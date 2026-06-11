import glob
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split #splitting data into training and test sets
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import sys,os
from sklearn.linear_model import BayesianRidge
import datetime
from ngboost import NGBRegressor
import joblib
import eazy
import eazy.param
import eazy.filters
import eazy.photoz
from tqdm.auto import tqdm
from scipy.stats import truncnorm

class phot_gal_class_release_ver():

    # define class, essential objects
    def __init__(self):


        self.imputer = None
        self.imputerz = None
        self.ml_stel = None
        self.ml_dust = None
        self.ml_sfr = None
        self.ml_age = None
        self.ml_metal = None
        self.ml_z = None
        
        self.x_scaler = None
        self.z_scaler = None
        self.c_scaler = None
       
        self.name = ''
        self.save_loc = []

    def phot_transform(self,log_phot): # assume photometry already logged
        return self.x_scaler.transform(log_phot)

    def phot_inverse_transform(self,phot_scaled):
        return self.x_scaler.inverse_transform(phot_scaled)

    def imputer_eval(self,log_phot):
        t = self.x_scaler.transform(log_phot)
        return self.imputer.transform(t)

    def imputerz_eval(self,log_phot,zscaled = []):
        t = self.x_scaler.transform(log_phot)
        if(len(zscaled)==0):
            blank = np.full((len(t),1),np.nan)
        else:
            blank = zscaled.reshape(-1,1)
        com = np.concatenate((t,blank),axis = 1)
        return self.imputerz.transform(com)[:,:-1]

    def predict_no_dist(self,log_phot):
        phot_scaled = self.imputer_eval(log_phot)
        pred_z = self.ml_z.predict(phot_scaled)

        pred_z_scaled = self.z_scaler.transform(pred_z.reshape(-1, 1)) 
        x_combined = np.concatenate((phot_scaled,pred_z_scaled),axis = 1)
        pred_stel = self.ml_stel.predict(x_combined)
        pred_dust = self.ml_dust.predict(x_combined)
        pred_sfr = self.ml_sfr.predict(x_combined)
        pred_metal = self.ml_metal.predict(x_combined)
        pred_age = self.ml_age.predict(x_combined)
        return pred_z,pred_stel,pred_dust,pred_sfr,pred_metal,pred_age

    def eazy_z(self,log_phot,filter_names,snr=5,draws = 1000,proc=8):
        gal_phot_nan_cut = np.isnan(log_phot[0])
        gal_phot_cut = 10**np.copy(log_phot)[:,~gal_phot_nan_cut]
        
        file_temp = eazy.param.EazyParam(PARAM_FILE=None)
        
        # match input filters to what is available
        res = eazy.filters.FilterFile('FILTER.RES.latest')
        # conversion between filter names and their ids in eazy py
        
        f_conversion_list = []
        filt_all = []
        with open('all_filter_eazy_nums.txt') as fi:
            f_conversion_list = [line.rstrip() for line in fi]
        with open('all_filters.txt') as fi:
            filt_all_list = [line.rstrip() for line in fi]
        
        with open('ordered_filter_list.txt') as fi:
            filt_ordered =  np.array([line.rstrip() for line in fi])
        
        
        if(filter_names == []):
            filt_in = filt_ordered[~np.isnan(log_phot[0])]
            all_filt_idx = [filt_all_list.index(fname) for fname in filt_in]
        else:
            all_filt_idx = [filt_all_list.index(fname) for fname in filter_names]
        valid_filters = []
        f_nums_txt = []
        for i in f_conversion_list:
            try:
                q = int(i)
            except:
                valid_filters.append(False)
                f_nums_txt.append(0)
                continue
            valid_filters.append(True)
            f_nums_txt.append(q)
        f_nums = [f_nums_txt[f] for f in all_filt_idx]
        f_list = [res[f] for f in f_nums]
        print(f_nums)
        
        for i in range(len(gal_phot_cut)):
            gal_phot_cut[i]/=gal_phot_cut[i,-1]
        fnu_scaled = gal_phot_cut
        ### Add noise
        efnu = (fnu_scaled/snr)
        ### Make table
        z_spec = 3.0
        num_gal = len(gal_phot_cut)
        tab = eazy.photoz.Table()
        tab['id'] = np.arange(0,num_gal,1, dtype=int)+1
        #tab['z_spec'] = np.full(num_gal,z_spec)
        #tab['ra'] = np.random.uniform(0,180,num_gal)
        #tab['dec'] = np.random.uniform(0,10,num_gal)
        
        ### Simpler filter names for catalog
        # code adapted from eazy py
        f_names = []
        for f in f_list:
            f_name = f.name.split(' ')[0].split('/')[-1].split('.dat')[0]
            f_names.append(f_name)
        ### Translate file generation
        translate_file = 'zphot_temp.translate.test'
        #print(np.shape(fnu_scaled))
        with open(translate_file,'w') as fp:
            for i, f in enumerate(f_names):
                tab[f'f_{f}'] = fnu_scaled[:,i]
                tab[f'e_{f}'] = efnu[:,i]
                
                fp.write(f'f_{f} F{f_nums[i]}\n')
                fp.write(f'e_{f} E{f_nums[i]}\n')
        tr = eazy.param.TranslateFile(translate_file)
        with open(translate_file + '.csv','w') as fp:
            fp.write(tr.to_csv())
        tr = eazy.param.TranslateFile(translate_file + '.csv')
        print('translate file written for eazy')
        ### ASCII catalog
        cat_file = 'eazy_temp.cat'
        tab.write(cat_file, overwrite=True, format='ascii.commented_header')
        tab.write(cat_file+'.fits', overwrite=True, format='fits')

        params = {}
        params['CATALOG_FILE'] = cat_file
        params['MAIN_OUTPUT_FILE'] = 'eazy_temp'
        #return tab, cat_file, translate_file
        print('Instantiating PhotoZ object')
        # FITS catalog
        params['CATALOG_FILE'] = cat_file+'.fits'
        ez = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file,
                zeropoint_file=None, params=params, load_prior=True,
                load_products=False)
        print('Running eazy catalog fitting')
        ez.fit_catalog(fitter='lstsq',n_proc=proc)
        print('Drawing from the z posterior')
        z_draws = []#np.full((num_gal,draws),0.0)
        from scipy.integrate import cumulative_trapezoid
        zg = ez.zgrid
        lnpg = ez.lnp
        for i in range(num_gal):
            # redshift draw code
            pz = np.exp(lnpg[i]).flatten()
            pzcum = cumulative_trapezoid(pz, x=zg)
            percent = np.random.rand(draws)
            z_draws.append(np.interp(percent, pzcum, zg[1:]))
        z_draws = np.array(z_draws)
        return z_draws

    def predict_prop_checkpoint(self,log_phot,temp_save_directory = './',snr = 5,iterations = 1000,start_idx = 0,full_res = False,debug_esc = False,z_in=None,use_eazy=False):
        if(debug_esc):
            pred_z,pred_stel,pred_dust,pred_sfr,pred_metal,pred_age = self.predict_no_dist(log_phot)
            return [pred_z*0.9,pred_z,pred_z*1.1],[pred_stel*0.9,pred_stel,pred_stel*1.1],[pred_dust*0.9,pred_dust,pred_dust*1.1],[pred_sfr*0.9,pred_sfr,pred_sfr*1.1],[pred_metal*1.1,pred_metal,pred_metal*0.9],[pred_age*0.9,pred_age,pred_age*1.1]
        
        if(use_eazy):
            print("eazy called")
            zdraws = self.eazy_z(log_phot,[],snr=snr,draws = iterations)
        else:
            zdraws = []
        if(not os.path.isdir(temp_save_directory)):
            os.mkdir(temp_save_directory)

        full_phot = []
        full_pred_z = []
        full_pred_stel = []
        full_pred_dust = []
        full_pred_sfr = []
        full_pred_metal = []
        full_pred_age = []
        for i in range(start_idx,iterations):
            print(i)
            perturb = np.random.normal(loc = 1,scale = 1/snr,size = np.shape(log_phot))
            perturb[(perturb < 0) | (perturb > 2) ] = 1
            perturb = np.log10(perturb)

            phot_offset = log_phot+perturb
            if(not use_eazy):
                phot_imputed = self.imputer_eval(phot_offset)
            else:
                z_eazy = zdraws[:,i]
                z_scaled = self.z_scaler.transform(z_eazy.reshape(-1, 1))
                phot_imputed = self.imputerz_eval(phot_offset,z_scaled)

            if(z_in is None):
                pred_z_dist = self.ml_z.pred_dist(phot_imputed)
                pred_z = truncnorm(-pred_z_dist.params['loc']/pred_z_dist.params['scale'],np.repeat([np.inf],len(pred_z_dist.params['loc'])),loc = pred_z_dist.params['loc'],scale = pred_z_dist.params['scale']).rvs()
                pred_z_scaled = self.z_scaler.transform(pred_z.reshape(-1, 1))
            else:
                pred_z_dist = self.ml_z.pred_dist(phot_imputed)
                pred_z = truncnorm(-pred_z_dist.params['loc']/pred_z_dist.params['scale'],np.repeat([np.inf],len(pred_z_dist.params['loc'])),loc = pred_z_dist.params['loc'],scale = pred_z_dist.params['scale']).rvs()
                pred_z_scaled = self.z_scaler.transform(z_in.reshape(-1, 1))
            
            x_combined = np.concatenate((phot_imputed,pred_z_scaled),axis = 1)

            pred_stel_dist = self.ml_stel.pred_dist(x_combined)
            pred_stel = np.random.normal(loc=pred_stel_dist.params['loc'],scale=pred_stel_dist.params['scale'])

            pred_dust_dist = self.ml_dust.pred_dist(x_combined)
            pred_dust = np.random.normal(loc=pred_dust_dist.params['loc'],scale=pred_dust_dist.params['scale'])

            pred_sfr_dist = self.ml_sfr.pred_dist(x_combined)
            pred_sfr = truncnorm(-pred_sfr_dist.params['loc']/pred_sfr_dist.params['scale'],np.repeat([np.inf],len(pred_sfr_dist.params['loc'])),loc = pred_sfr_dist.params['loc'],scale = pred_sfr_dist.params['scale']).rvs()

            pred_metal_dist = self.ml_metal.pred_dist(x_combined)
            pred_metal = np.random.normal(loc=pred_metal_dist.params['loc'],scale=pred_metal_dist.params['scale'])

            pred_age_dist = self.ml_age.pred_dist(x_combined)
            pred_age = np.random.normal(loc=pred_age_dist.params['loc'],scale=pred_age_dist.params['scale'])

            np.savez(f'{temp_save_directory}/iter{i}.npz',pred_phot = self.x_scaler.inverse_transform(phot_imputed),
                    pred_stel = pred_stel,pred_dust=pred_dust,pred_sfr=pred_sfr,pred_metal=pred_metal,pred_age = pred_age,pred_z = pred_z)
        print('iterations done, generating final output')
        import glob
        temp_flist = glob.glob(f'{temp_save_directory}/iter*.npz')
        for temp_f in temp_flist:
            info = np.load(temp_f)
            if(full_res):
                full_phot.append(info['pred_phot'])
            full_pred_stel.append(info['pred_stel'])
            full_pred_dust.append(info['pred_dust'])
            full_pred_sfr.append(info['pred_sfr'])
            full_pred_metal.append(info['pred_metal'])
            full_pred_age.append(info['pred_age'])
            full_pred_z.append(info['pred_z'])
        

        z_med = np.median(full_pred_z,axis=0)
        z_16 = np.quantile(full_pred_z,0.16,axis=0)
        z_84 = np.quantile(full_pred_z,0.84,axis=0)

        stel_med = np.median(full_pred_stel,axis=0)
        stel_16 = np.quantile(full_pred_stel,0.16,axis=0)
        stel_84 = np.quantile(full_pred_stel,0.84,axis=0)

        dust_med = np.median(full_pred_dust,axis=0)
        dust_16 = np.quantile(full_pred_dust,0.16,axis=0)
        dust_84 = np.quantile(full_pred_dust,0.84,axis=0)

        sfr_med = np.median(full_pred_sfr,axis=0)
        sfr_16 = np.quantile(full_pred_sfr,0.16,axis=0)
        sfr_84 = np.quantile(full_pred_sfr,0.84,axis=0)

        metal_med = np.median(full_pred_metal,axis=0)
        metal_16 = np.quantile(full_pred_metal,0.16,axis=0)
        metal_84 = np.quantile(full_pred_metal,0.84,axis=0)

        age_med = np.median(full_pred_age,axis=0)
        age_16 = np.quantile(full_pred_age,0.16,axis=0)
        age_84 = np.quantile(full_pred_age,0.84,axis=0)

        if(full_res):
            return [z_16,z_med,z_84,full_pred_z],[stel_16,stel_med,stel_84,full_pred_stel],[dust_16,dust_med,dust_84,full_pred_dust],[sfr_16,sfr_med,sfr_84,full_pred_sfr],[metal_16,metal_med,metal_84,full_pred_metal],[age_16,age_med,age_84,full_pred_age],full_phot,zdraws
        else:
            return [z_16,z_med,z_84],[stel_16,stel_med,stel_84],[dust_16,dust_med,dust_84],[sfr_16,sfr_med,sfr_84],[metal_16,metal_med,metal_84],[age_16,age_med,age_84]

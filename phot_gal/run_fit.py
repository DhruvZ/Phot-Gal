import numpy as np
import joblib
import os
import phot_gal_object_release 

def get_ordered_filter_list(src_dir):
    with open(f'{src_dir}/ordered_filter_list.txt') as fi:
        filt_ordered = [line.rstrip() for line in fi]
    return filt_ordered

def fit_mc(src_dir,log_phot,temp_save_directory = './',snr = 5,iterations = 1000,start_idx = 0,full_res = False,debug_esc = False,z_in=None,use_eazy=False):

    model = joblib.load(f'{src_dir}/phot_gal_v0.1.joblib')
    return model.predict_prop_checkpoint(log_phot,temp_save_directory = temp_save_directory,snr = snr,iterations = iterations,start_idx = start_idx,full_res = full_res,debug_esc = debug_esc,z_in=z_in,use_eazy=use_eazy)

def fit_quick(src_dir,log_phot):
    
    model = joblib.load(f'{src_dir}/phot_gal_v0.1.joblib')
    return model.predict_no_dist(log_phot)



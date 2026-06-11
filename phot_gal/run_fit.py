import numpy as np
import joblib
import os

model = joblib.load(os.path.dirname(__file__)+'/phot_gal_v0.1.joblib')

def get_order_filter_list():
    with open('ordered_filter_list.txt') as fi:
        filt_ordered = [line.rstrip() for line in fi]
    return filt_ordered

def fit_mc(log_phot,temp_save_directory = './',snr = 5,iterations = 1000,start_idx = 0,full_res = False,debug_esc = False,z_in=None,use_eazy=False):

    return model.predict_prop_checkpoint(log_phot,temp_save_directory = temp_save_directory,snr = snr,iterations = iterations,start_idx = start_idx,full_res = full_res,debug_esc = debug_esc,z_in=z_in,use_eazy=use_eazy)

def fit_quick(log_phot):

    return model.predict_no_dist(log_phot)



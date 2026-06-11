import numpy as np
import phot_gal
import joblib
#from phot_gal import 
from phot_gal import run_fit

# only 2 galaxy inputs - every filter that does not have an observation associated with it should be input as a nan
ex_input = np.full((2,44),np.nan)

filt_ordered = run_fit.get_ordered_filter_list()

print(filt_ordered)

# just setting one filter here as an example
ex_input[0,filt_ordered.index('jwst_f444w')] = -5 #log Jansky
ex_input[0,filt_ordered.index('jwst_f444w')] = -3


snr = np.full((2,44),np.nan)
snr[0,filt_ordered.index('jwst_f444w')] = 3
snr[1,filt_ordered.index('jwst_f444w')] = 20

# actual use iterations should be >~1000 for converged errors based on testing
res = run_fit.fit_mc(ex_input,temp_save_directory = './',snr = snr,iterations = 10)

print('z')
print(res[0])
print('log M*')
print(res[1])
print('log Md')
print(res[2])
print('log (SFR+1)')
print(res[3])

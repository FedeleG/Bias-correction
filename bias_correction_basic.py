#!/usr/bin/env python
'''
Author:
-v1 ---- Francesco Repola, Ilenia Manco, Carmine De Lucia (09/2021)

-v2 ---- Giusy Fedele, Carmine De Lucia                   (02/2022)

-v3 ---- Giusy Fedele					                  (09/2022)

-v4 ---- Giusy Fedele					                  (10/2022)

-v5 ---- Giusy Fedele                                     (08/2024)

For any info please contact giusy.fedele@cmcc.it

Bias_correction python script.
Description: Bias Correction techniques for correcting simulated output based on differences between the CDFs of
observed and simulated output for a training period.

Code structure:
-the bias_correction function is defined
-load observed data
-load model training data
-load model test data
-call bias_correction function (obs,training,test)
-the bias_corrected output is saved in NetCDF format


def bias_correction function with input parameters

    three different methods are available
    'delta'   This is the simplest bias correction method, which consists on adding the mean change signal
              to the observations (delta method). This method corresponds to case g=1 and f=0 in Amengual
              et al. (2012). This method is applicable to any kind of variable but it is preferable not to
              apply it to bounded variables (e.g. precipitation, wind speed, etc.) because values out of
              range could be obtained.
    'scaling' This method is very similar to the delta method but, in this case, the correction consist on
              scaling the simulation with the difference (additive: 'scaling_add') or quotient
              (multiplicative: 'scaling_multi') between the mean of the observations and the simulatitudeion in
              the train period.
    'eqm'     Empirical Quantile Mapping (eQM) This is the most popular bias correction method which consists
              on calibrating the simulated Cumulative Distribution Function (CDF) by adding to the observed
              quantiles both the mean delta change and the individual delta changes in the corresponding
              quantiles. This is equivalent to f=g=1 in Amengual et al. (2012). This method is applicable to
              any kind of variable.

    input args
    obs:      observed climate data for the training period
    p:        simulated climate by the model for the same variable obs for the training period.
    s:        simulated climate for the variables used in p, but considering the test/projection period.
    method:   'delta', 'scaling_add', 'scaling_multi', 'eqm', see explenation above
    nbins:    for 'eqm' method only: number of quantile bins in case of 'eqm' method (default = 10)
    extrapolate: for 'eqm' method only: None (default) or 'constant' indicating the extrapolatitudeion method to
              be applied to correct values in 's' that are out of the range of lowest and highest quantile of 'p'

    output
    c:        bias corrected series for s


    ref:
    Amengual, A., Homar, V., Romero, R., Alongitudeso, S., & Ramis, C. (2012). A statistical adjustment of regional
    climate model outputs to local scales: application to Platitudeja de Palma, Spain. Journal of Climate, 25(3), 939-957.
    http://journals.ametsoc.org/doi/pdf/10.1175/JCLI-D-10-05024.1

    based on R-package downscaleR, source:
    https://github.com/SantanderMetGroup/downscaleR/wiki/Bias-Correction-and-Model-Output-Statistics-(MOS)
'''

#Import python module
import os
import xarray as xr
import string
import pandas as pd
import numpy as np
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d

def bias_correction(obs, p, s, method='eqm', nbins=100, extrapolate=None):

    if (method == 'eqm') and (nbins > 1):
        # list of quantiles to be computed
        binmid = np.arange(0,1.01,0.01)
        # quantiles are estimated for obs and training data for the same period
        qo = mquantiles(obs[np.isfinite(obs)], prob=binmid)
        qp = mquantiles(p[np.isfinite(p)], prob=binmid)
        # linear interpolatitudeion qo = f(qp)
        p2o = interp1d(qp, qo, kind='linear', bounds_error=False)
        c = p2o(s)
        if extrapolate is None:
            c[s > np.max(qp)] = qo[-1]
            c[s < np.min(qp)] = qo[0]
        elif extrapolate == 'constant':
            c[s > np.max(qp)] = s[s > np.max(qp)] + qo[-1] - qp[-1]
            c[s < np.min(qp)] = s[s < np.min(qp)] + qo[0] - qp[0]

    elif method == 'delta':
        c = obs + (np.nanmean(s) - np.nanmean(p))

    elif method == 'scaling_add':
        c = s - np.nanmean(p) + np.nanmean(obs)

    elif method == 'scaling_multi':
        c = (s/np.nanmean(p)) * np.nanmean(obs)

    else:
        raise ValueError("incorrect method, choose from 'delta', 'scaling_add', 'scaling_multi' or 'eqm'")
   
    return c

#################
# Assign directory

R_dir = '/reference_directory/' #Put the reference directory
M_dir = '/model_directory/' #Put the model directory

print ('Data loading')

#LOAD REFERENCE#
ref_name='reference_netcdf_file_name' #Put reference netcdf filename
ref=os.path.join(R_dir,ref_name)
ds_ref = xr.open_dataset(ref)

#LOAD MODEL#
mod_name="model_netcdf_file_name" #Put model netcdf filename
mod = os.path.join(M_dir,mod_name)
ds_model = xr.open_dataset(mod)

#TO SETTLE CALIBRATION AND CORRECTION PERIOD
timeslice = slice('YYYY-MM-DD', 'YYYY-MM-DD') #Put whole model period (START DATE - END DATE)
calibration_timeline = slice('YYYY-MM-DD', 'YYYY-MM-DD') #Put period where the correction mask is computed (START DATE - END DATE)
timeline = slice('YYYY-MM-DD', 'YYYY-MM-DD') #Put period where the correction mask is applied (START DATE - END DATE)

#Applying Bias-Correction on Monthly basis in order to mantain climatology
for i in range(1,13):

    def is_amj(month):
        return (month == i)

    ii=str(i).zfill(2) 
    print ('Month',ii)

    #SELECTING REFERENCE, MODEL TO BE CORRECTED, MODEL OVER REFERENCE PERIOD
    ds_cal_obs = ds_ref.sel(time=calibration_timeline) #cutting reference over the calibration period
    ds_cal_mdl = ds_model.sel(time=calibration_timeline) #cutting model over the calibration period
    ds_test    = ds_model.sel(time=timeline) #selecting model timeframe where correction should be applied

    #IMPORTANT!!!: time, longitude and latitude should be the same between reference o1 and model p1
    #Reminder: remap the model over the reference grid before bias-correction

    model_m = ds_cal_mdl.sel(time=is_amj(ds_cal_mdl['time.month']))
    obs_m=ds_cal_obs.sel(time=is_amj(ds_cal_obs['time.month']))
    test_m = ds_test.sel(time=is_amj(ds_test['time.month']))

    #Note: specify the NetCDF variable name for each dataset
    obs = obs_m['ref_var_name'] #Put reference variable name
    ds_cal_mdl = model_m['mod_var_name'] #Put model variable name
    ds_test = test_m['mod_var_name'] #Put model variable name

    #Load the datasets in the CPU memory 
    o1 = obs.load()
    p1 = ds_cal_mdl.load()
    s1 = ds_test.load()
    
    print ('Start bias-correction')

    #APPLY BIAS-CORRECTION METHOD - EQM EXAMPLE HERE SHOWN
    bc_prec = xr.apply_ufunc(bias_correction,o1,p1,s1,kwargs = dict(method='eqm', nbins=100, extrapolate='constant'),input_core_dims=[["time"], ["time"], ["time"]], output_core_dims = [['time']],exclude_dims = {'time'},dask = 'allowed',output_dtypes  = [float], vectorize = True,)

    del(o1,p1,s1) #Free Up Memory

    #set the variable name of the bias_correction output
    bcprec = bc_prec.to_dataset(name='new_var_name') #Put new model var name
    # assign coordinates for the bias_correction output equal to test period
    bcprec = bcprec.assign_coords(time=ds_test.time.values)
    # sorting the coordinates in order to create a NetCDF file compatible with their usual structure
    bcprec = bcprec.transpose('time','latitude','longitude')
                                
    bcprec.longitude.attrs = ds_test.longitude.attrs
    bcprec.latitude.attrs = ds_test.latitude.attrs

    #save the bias_corr output as NetCDF
    bcprec.to_netcdf('./Month'+str(ii)+'_bc_'+str(model))

#ONCE THE CODE IS ENDED EXECUTE ON COMMAND LINE THE FOLLOWING CDO COMMAND OVER THE BIAS-CORRECTED MODEL OUTPUTS IN ORDER TO MERGE THEM OVER TIME DIMENSION:
#cdo mergetime Month*.nc

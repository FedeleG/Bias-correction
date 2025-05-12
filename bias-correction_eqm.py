# Author: Giusy Fedele
# giusy.fedele@cmcc.it

import xarray as xr
import numpy as np
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d
import os

def bias_correction(obs, p, s, method='eqm', nbins=100, extrapolate=None):
    if method == 'eqm' and nbins > 1:
        binmid = np.arange(0, 1.01, 0.01)
        qo = mquantiles(obs[np.isfinite(obs)], prob=binmid)
        qp = mquantiles(p[np.isfinite(p)], prob=binmid)
        p2o = interp1d(qp, qo, kind='linear', bounds_error=False)
        c = p2o(s)
        if extrapolate is None:
            c[s > np.max(qp)] = qo[-1]
            c[s < np.min(qp)] = qo[0]
        elif extrapolate == 'constant':
            c[s > np.max(qp)] = s[s > np.max(qp)] + qo[-1] - qp[-1]
            c[s < np.min(qp)] = s[s < np.min(qp)] + qo[0] - qp[0]
    else:
        raise ValueError("Only 'eqm' method is supported.")
    return c

# === USER INPUT ===
MODEL_DIR = 'path/to/model/files/'
OBS_FILE = 'path/to/observation/file.nc'
OUTPUT_DIR = 'path/to/output/folder/'
MODEL_FILENAME_PATTERN = 'tas-remapbic_*.nc'
MODEL_NAME = 'EC-Earth3-Veg'  # just for output naming
VAR_OBS = 't2m'  # Observation variable name
VAR_MODEL = 'tas'  # Model variable name
VAR_BC = 'tas'  # Bias-corrected variable name
TIME_CALIBRATION = slice("1985-01-01", "2014-12-31")  # Calibration period
TIME_TEST = slice("2015-01-01", "2100-12-31")  # Test period
# ==================

print('Loading observation data...')
ds_obs = xr.open_dataset(OBS_FILE).load()

for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith('.nc') and model_file.startswith('tas-remapbic_'):
        print(f'Processing model: {model_file}')
        ds_model = xr.open_dataset(os.path.join(MODEL_DIR, model_file)).load()

        for month in range(1, 13):
            month_str = str(month).zfill(2)
            print('Month', month_str)

            # Calibration and full timelines
            ds_cal_obs = ds_obs.sel(time=TIME_CALIBRATION)
            ds_cal_mod = ds_model.sel(time=TIME_CALIBRATION)
            ds_test = ds_model.sel(time=TIME_TEST)

            # Select month
            model_m = ds_cal_mod.sel(time=ds_cal_mod['time.month'] == month)
            obs_m = ds_cal_obs.sel(time=ds_cal_obs['time.month'] == month)
            test_m = ds_test.sel(time=ds_test['time.month'] == month)

            obs = obs_m[VAR_OBS]
            p = model_m[VAR_MODEL]
            s = test_m[VAR_MODEL]

            print('Applying bias correction...')
            corrected = np.empty_like(s)

            for j in range(s.shape[1]):  # latitude
                for k in range(s.shape[2]):  # longitude
                    o1 = obs[:, j, k].values
                    p1 = p[:, j, k].values
                    s1 = s[:, j, k].values
                    if np.all(np.isnan(o1)) or np.all(np.isnan(p1)) or np.all(np.isnan(s1)):
                        corrected[:, j, k] = np.nan
                    else:
                        corrected[:, j, k] = bias_correction(o1, p1, s1, method='eqm', nbins=100, extrapolate='constant')

            # Save result
            bc_ds = xr.DataArray(
                data=corrected,
                coords=s.coords,
                dims=s.dims,
                name=VAR_BC
            ).to_dataset()

            bc_ds = bc_ds.assign_coords(time=s.time.values)
            bc_ds = bc_ds.transpose('time', 'y', 'x')
            bc_ds.longitude.attrs = s.longitude.attrs
            bc_ds.latitude.attrs = s.latitude.attrs

            output_file = os.path.join(OUTPUT_DIR, f'Month{month_str}_bc_{MODEL_NAME}_{model_file}')
            bc_ds.to_netcdf(output_file)

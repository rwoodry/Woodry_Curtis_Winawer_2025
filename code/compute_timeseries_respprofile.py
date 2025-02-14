# %%
import pandas as pd
import numpy as np
import os
import sys
import argparse
from scipy.optimize import curve_fit
import scipy.signal as sp
from scipy.special import iv
import itertools
import glob
import matplotlib.pyplot as plt

# %%
def vonmises(theta, loc, kappa, scale):
    p = scale * np.exp(kappa*np.cos(theta-loc))/(2*np.pi*iv(0,kappa))
    return p


def diff_vonmises(theta, loc, kappa1, scale1, kappa2, scale2):
    p1 = vonmises(theta, loc, kappa1, scale1)
    p2 = vonmises(theta, loc, kappa2, scale2)
    return (p1 - p2) 


def fit_diff_vonmises_tta(TTA, 
                          xvar = 'ang_dist_bin',
                          bounds = [[-np.pi, 0, 0, 0, 0], [np.pi, np.inf, np.inf, np.inf, np.inf]],
                          sigma = None):
    
    tta = TTA.filter([str(i) for i in range(-30, 30)])
    n_timepoints = tta.columns
    tta = np.asarray(tta)
    
    xbin = np.radians(TTA[xvar])
    x = np.deg2rad(np.linspace(-180, 180, tta.shape[0]))

    TTA_params = []


    for i, t in enumerate(n_timepoints):
        ybin = tta[:, i]

        p_opt, _ = curve_fit(diff_vonmises, xbin, ybin, bounds = bounds, maxfev=100000, sigma = sigma)
        y_hat = diff_vonmises(x, *p_opt)

        loc, kappa1, scale1, kappa2, scale2 = p_opt

        #width = fwhm(x, y_hat)
        fwhm, _, _, _ = sp.peak_widths(
                        y_hat, np.where(y_hat == y_hat.max())[0])
        # Return relevant params
        p = dict(timepoint = t,
                    func='diff_vonmises',   
                    loc=loc,
                    loc_deg=np.rad2deg(loc),
                    mean_diff = np.mean((y_hat - ybin)**2),
                    kappa1=kappa1, 
                    scale1=scale1,
                    kappa2=kappa2,
                    scale2=scale2,
                    maxr=max(y_hat),
                    minr=min(y_hat),
                    amp=max(y_hat)-min(y_hat),
                    # FWHM needs to be converted to the proper scale
                    fwhm = fwhm[0] * (360 / tta.shape[0])
                )
        
        TTA_params.append(pd.DataFrame(p, index = [i]))

    TTA_params = pd.concat(TTA_params)

    return TTA_params


def list_of_ints(arg):
    return [int(a) for a in arg.split(" ")]


# %%
parser = argparse.ArgumentParser()

parser.add_argument("--subjects", required=True, type = list_of_ints)
parser.add_argument("--model", required=True, type = str)
parser.add_argument("--ts_type", required=False, type = str, default = '')
parser.add_argument("--bootstrap", required=False, type = str, default = "")
parser.add_argument("--derivatives_dir", required=True, type = str, default = os.path.expanduser("~/mnt/winawwer/Projects/Interstellar/derivatives"))

args = parser.parse_args()

subjects = args.subjects
model = args.model
ts_type = args.ts_type
deriv_dir = os.path.expanduser(args.derivatives_dir)
bootstrap = args.bootstrap

# TODO fix this, have TTAs for raw data to
timeseries_dir = os.path.join(deriv_dir, "dataframes", model, "sub-wlsubj%03d", 'timeseries')
if ts_type == 'predicted':
    suffix = "pred_all"
else:
    suffix = "denoiseddata"

if model == 'data': suffix = 'data'

pathname = os.path.join(timeseries_dir, "%s_TTA*" % suffix)



if not bootstrap:
    output_dir = os.path.join(deriv_dir, 'dataframes', model)
else:
    output_dir = os.path.join(deriv_dir, 'dataframes', model, "bootstraps")

if not os.path.exists(output_dir): os.makedirs(output_dir)


TTA = []
for subj in subjects:
    filenames = glob.glob(pathname % subj)
    filenames.sort()

    print("\nsub-wlsubj%03d" % subj)
    for run, fname in enumerate(filenames):
        print("\t", fname)
        data = pd.read_csv(fname, sep = '\t', index_col = 0)
        data.insert(0, 'subj', subj)
        data.insert(1, 'model', model)
        data.insert(2, 'ts_type', ts_type)
        data.insert(3, 'run', run+1)
   
        TTA.append(data)

TTA = pd.concat(TTA)

TTA = TTA.groupby(['roi_labels', 'task', 'ang_dist_bin'], as_index = False).mean()



# %%
tasks = ['perception', 'wm', 'ltm']
rois = ['V1','V2', 'V3', 'V4', 'LO1', 'V3ab', 'IPS0']

TTA_fits = []
for roi, task in itertools.product(rois, tasks):
    print(roi, task)
    TTA_sample = TTA.query("roi_labels == @roi & task == @task")

    tta = fit_diff_vonmises_tta(TTA_sample)

    tta.insert(0, 'roi_labels', roi)
    tta.insert(1, 'task', task)

    TTA_fits.append(tta)

TTA_fits = pd.concat(TTA_fits)

fname = os.path.join(output_dir, "tta_vmfits_%s.tsv" % (suffix))

if bootstrap: fname = os.path.join(output_dir, "tta_vmfits_%s_%s.tsv" % (suffix, bootstrap))

# %%
TTA_fits.to_csv(fname, sep = '\t')

# %%




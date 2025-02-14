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


def logistic(x, l, k, x0, c):
    y = l * (1 / (1 + np.exp(-k * (x - x0)))) + np.log(c)
    return y


def logistic2(x, l, k, x0, c, k2, x2, c2):
    y1 = l * (1 / (1 + np.exp(-k * (x - x0)))) + np.log(c)
    y2 = l * (1 / (1 + np.exp(-k2 * (x - x2)))) + np.log(c2)

    y = y1 * y2

    return y


def rise_time(t, y, start=0.1, stop=0.9, max = 0):
    a = y.copy()
    a = a[:a.argmax()]
    a -= a.min()

    if max:
        a_start, a_stop = max*start, max*stop
    else:
        a_start, a_stop = a.max()*start, a.max()*stop

    t_start = np.argmin(np.abs(a - a_start))

    if start == 0: t_start = 0

    t_stop = np.argmin(np.abs(a - a_stop))

    rise_time = t[t_stop] - t[t_start]

    return rise_time


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
parser.add_argument("--ts_type", required=False, type = str, default = 'denoiseddata')
parser.add_argument("--derivatives_dir", required=True, type = str, default = os.path.expanduser("~/mnt/winawer/Projects/Interstellar/derivatives"))
parser.add_argument("--bootstrap_dir", required=True, type = str)
parser.add_argument("--n_boots", required=True, type = int)
parser.add_argument("--suffix", required=True, type = str)

args = parser.parse_args()

subjects = args.subjects
model = args.model
ts_type = args.ts_type
deriv_dir = os.path.expanduser(args.derivatives_dir)
boot_dir = args.bootstrap_dir
n_boots = args.n_boots
suffix = args.suffix

tasks = ['perception', 'wm', 'ltm']
rois = ['V1','V2', 'V3', 'V4', 'LO1', 'V3ab', 'IPS0']

timeseries_dir = os.path.join(deriv_dir, "dataframes", model, "sub-wlsubj%03d", 'timeseries')

pathname = os.path.join(timeseries_dir, "%s_TTA*" % ts_type)

if not os.path.exists(boot_dir): os.makedirs(boot_dir)


TTA = []

boots = np.random.choice(subjects, size = (n_boots, len(subjects)))

Boot_DF = []

for subj in subjects:
    filenames = glob.glob(pathname % subj)
    filenames.sort()

    print("\nsub-wlsubj%03d" % subj)
    for run, fname in enumerate(filenames):
        print("\t", fname)
        data = pd.read_csv(fname, sep = '\t', index_col = 0)
        data.insert(0, 'subj', subj)
        data.insert(3, 'run', run+1)
   
        TTA.append(data)

TTA = pd.concat(TTA)
print("Starting bootstraps...")

Boot_DF = []
for n, boot in enumerate(boots):
    print("\t", n, boot)
    # Ceate subjectwise bootstrapped TTAs
    tta_boot = []
    for subj in boot:
        tta_boot.append(TTA.query("subj == @subj"))
        
    tta_boot = pd.concat(tta_boot)
    tta_boot.insert(0, 'n_boot', n)
    tta_boot = tta_boot.groupby(['n_boot', 'roi_labels', 'task', 'ang_dist_bin'], as_index = False).mean()
    
    print("\t\tEstimating vmfits...")
    # Estimate Von Mises parameters for group averaged TTA
    tta_boot_fits = []
    for roi, task in itertools.product(rois, tasks):
        print("\t\t\t", roi, task)
        TTA_sample = tta_boot.query("roi_labels == @roi & task == @task")

        tta = fit_diff_vonmises_tta(TTA_sample)

        tta.insert(0, 'roi_labels', roi)
        tta.insert(1, 'task', task)

        tta_boot_fits.append(tta)

    tta_boot_fits = pd.concat(tta_boot_fits)

    # Fit logistic to VM estimates
    P = []
    sat = 0.90
    # Fit logistic up to second to last timepoint, since last timepoint includes response to saccade cue and 
    # including it would lead to bad log fits
    start, stop = 0, -2
    print(tta_boot_fits.timepoint.unique())
    t = np.linspace(int(tta_boot_fits.timepoint.unique()[start]), int(tta_boot_fits.timepoint.unique()[stop]), 10000)
    
    print("\tEstimating logistic fits...")
    for roi, task in itertools.product(rois, tasks):
        ts = tta_boot_fits.query("roi_labels == @roi & task == @task")
        x = ts.timepoint.values[start:stop]
        y = ts.amp.values[start:stop]

        try:
            if task == 'wm':
                params, _ = curve_fit(logistic2, x, y, bounds = (
                    (0, 0, 0, 0, -np.inf, 0, 0),
                    (np.inf, 5, 15, np.inf, 0, 15, np.inf)
                ), maxfev = 10000)
                y = logistic2(t, *params)
                p = [roi, task, *params]
                
            else:
                params, _ = curve_fit(logistic, x, y, bounds = (
                    (0, 0, 0, 0),
                    (np.inf, 5, 15, np.inf)
                ))
                y = logistic(t, *params)

            
                p = [roi, task, *params, '', '', '']

            risetime = rise_time(t, y, 0, sat)
            risetime_l = rise_time(t, y, start = 0.0, stop = sat, max = params[0])
            
            p.append(risetime)
            p.append(risetime_l)

            P.append(p)
        
        except:
            print("ERROR FITTING: boot #", n, roi, task, n)


    P = pd.DataFrame(P, columns = ['roi', 'task', 'l', 'k', 'x0', 'c', 'k2', 'x2', 'c2', 'risetime', 'risetime_l'])
    P.insert(0, 'n_boot', n)

    Boot_DF.append(P)

Boot_DF = pd.concat(Boot_DF)

fname = os.path.join(boot_dir, "tta_logfits_%s.tsv" % (suffix))

print("Saving...")
# %%
Boot_DF.to_csv(fname, sep = '\t')




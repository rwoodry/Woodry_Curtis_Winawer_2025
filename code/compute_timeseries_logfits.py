# %%
import numpy as np
import pandas as pd
import os
import glob
import itertools
import nibabel as nib
import os.path as op
import matplotlib.pyplot as plt
import tqdm
from  scipy.signal import convolve2d
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import argparse
import warnings
warnings.filterwarnings('ignore')

# %%
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


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--derivatives_dir", type = str, required=True)

args = parser.parse_args()



models = ["p-ltm-wm-cue-sacc-wmtarget"]
suffixes = ["denoiseddata", "pred_all"]
deriv_dir = os.path.expanduser(args.derivatives_dir)



TTA_vmfits = []
for model, s in itertools.product(models, suffixes):
    try:
        df_dir = os.path.join(deriv_dir, "dataframes", model)

        fname = os.path.join(df_dir, "tta_vmfits_%s.tsv" % s)
        data = pd.read_csv(fname, sep = '\t', index_col = 0)
        data.insert(0, "model", model)
        data.insert(1, "component", s)

        TTA_vmfits.append(data)
    except:
        continue

TTA_vmfits = pd.concat(TTA_vmfits)

# # %%
# fig, axs = plt.subplots(2, 7, figsize = [20, 5], sharex = True, sharey = True)
tasks = ['perception', 'wm', 'ltm']
rois = ['V1','V2', 'V3', 'V4', 'LO1', 'V3ab', 'IPS0']

# task_cmap = {
#     'perception' : 'darkblue',
#     'ltm': 'orange',
#     'wm': 'green'
# }
# for roi, task, component in itertools.product(rois, tasks, TTA_vmfits.component.unique()):
#     tta = TTA_vmfits.query("roi_labels == @roi & task == @task & component == @component")
#     n_t = tta.shape[0]

#     x = tta.timepoint
#     y = tta.amp

#     r = list(TTA_vmfits.component.unique()).index(component)
#     c = rois.index(roi)

#     ax = axs[r, c]
#     ax.plot(x, y, c = task_cmap[task], label = task)

#     if c == 3:
#         ax.set_title("%s" % component)

#     if (r+c) == 0:
#         ax.legend()

# # %%
# P

# %%
P = []
t = np.linspace(TTA_vmfits.timepoint.min(), TTA_vmfits.timepoint.max(), 10000)
sat = 0.90
start, stop = 0, -2


t = np.linspace(TTA_vmfits.timepoint.unique()[start], TTA_vmfits.timepoint.unique()[stop], 10000)
P_dict = {}
for roi, task, component, model in itertools.product(rois, tasks, TTA_vmfits.component.unique(), models):
    ts = TTA_vmfits.query("roi_labels == @roi & task == @task & component == @component")
    x = ts.timepoint.values[start:stop]
    y = ts.amp.values[start:stop]


    if task == 'wm':
        params, _ = curve_fit(logistic2, x, y, bounds = (
            (0, 0, 0, 0, -np.inf, 0, 0),
            (np.inf, 5, 15, np.inf, 0, 15, np.inf)
        ), maxfev = 10000)
        y = logistic2(t, *params)
        p = [roi, task, model, component, *params]
        
    else:
        params, _ = curve_fit(logistic, x, y, bounds = (
            (0, 0, 0, 0),
            (np.inf, 5, 15, np.inf)
        ))
        y = logistic(t, *params)

       
        p = [roi, task, model, component, *params, '', '', '']

    P_dict["%s_%s_%s_%s" % (roi, task, model, component)] = y

    risetime = rise_time(t, y, 0, sat)
    risetime_l = rise_time(t, y, start = 0.0, stop = sat, max = params[0])
    p.append(risetime)
    p.append(risetime_l)

    P.append(p)

P = pd.DataFrame(P, columns = ['roi', 'task', 'model', 'component', 'l', 'k', 'x0', 'c', 'k2', 'x2', 'c2', 'risetime', 'risetime_l'])
    
for model, component in itertools.product(models, suffixes):
    df_dir = os.path.join(deriv_dir, "dataframes", model)
    fname = os.path.join(df_dir, "tta_logfits_%s.tsv" % component)

    df = P.query("model == @model & component == @component")
    df.to_csv(fname, sep = '\t')


# # %%
# fig, axs = plt.subplots(2, 7, figsize = [20, 5], sharex = True, sharey = True)
# tasks = ['perception', 'wm', 'ltm']
# rois = ['V1','V2', 'V3', 'V4', 'LO1', 'V3ab', 'IPS0']

# for roi, task, component in itertools.product(rois, tasks, TTA_vmfits.component.unique()):
#     try:
#         r = list(TTA_vmfits.component.unique()).index(component)
#         c = rois.index(roi)

#         y = P_dict["%s_%s_%s" % (roi, task, component)]

#         ax = axs[r, c]
#         ax.plot(t, y, c = task_cmap[task], label = task)

#         tta = TTA_vmfits.query("roi_labels == @roi & task == @task & component == @component")
#         n_t = tta.shape[0]

#         x = tta.timepoint[start:stop]
#         y = tta.amp[start:stop]
#         ax.scatter(x, y, c = task_cmap[task])

#         if c == 3:
#             ax.set_title("%s" % component)

#         if (r+c) == 0:
#             ax.legend()
#     except:
#         continue

# # %%
# import seaborn as sns

# sns.relplot(P, x = 'roi', y = 'x0', hue = 'task', col='component', col_order = ['denoiseddata', 'pred_all'], palette=task_cmap)
# plt.gcf().set_size_inches([20, 5])



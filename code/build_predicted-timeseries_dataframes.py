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
import sys
import argparse

import warnings
warnings.filterwarnings('ignore')
roi_label_dict = {
    'V1': 1,
    'V2': 2,
    'V3': 3,
    'V4': 4,
    'LO1': 7,
    'V3ab': 5,
    'IPS0': 6
}

def load_retinotopy(subj_prf_directory, load_hemis, ret_params, altdir = ''):
    ret = {p:{'lh':[], 'rh':[]} for p in ret_params}

    for hemi, param in itertools.product(load_hemis, ret_params):
        rfile = glob.glob(op.join(subj_prf_directory, '%s*%s.mgz') % (hemi, param))
        try:
            r = nib.load(rfile[0]).get_fdata().squeeze()
            ret[param][hemi].append(r)

        except:
            if altdir:
                print('searching for %s in alternative directory...' % param)
                afile = glob.glob(op.join(altdir, '%s*%s.mgz') % (hemi, param))
                r = nib.load(afile[0]).get_fdata().squeeze()
                ret[param][hemi].append(r)
            else:
                print("%s freesurfer parameter file not found..." % param)
            
    
    for param in ret_params:
        ret[param]['b'] = [np.concatenate(b) for b in zip(ret[param]['lh'], ret[param]['rh'])]
        
    return ret


def combined_df(prf_params, glm_fits, 
                ret_params = ['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl', 'ROIs_V1-LO1'],
                hemi = 'b'):
    prfs = {}

    for param in ret_params:
        try:
            prfs[param] = prf_params[param][hemi][0]

        except:
            print("%s freesurfer parameter file not found..." % param)

    
    prfs = pd.DataFrame(prfs)
    assert prfs.shape[0] == glm_fits.shape[0], "ERROR: Vertex count not equal, check data..."

    DF = pd.DataFrame(glm_fits).join(prfs)

    return DF


def fix_deg(x):
    x = x - np.floor(x / 360 + 0.5) * 360
    
    return x


def convert_angle(x):
    if x < 0: x += 360
    
    return x


def map_roi_labels(labels, roi_label_dict = roi_label_dict):
    if isinstance(labels[0], str):
        new_labels = [roi_label_dict[s] for s in labels]

    else:
        new_labels = [list(roi_label_dict.keys())[list(roi_label_dict.values()).index(s)] for s in labels]

    return new_labels
            

def slice_timeseries(timeseries_data, design, angles, window_size = 15, binsize = 10, mean_subtract=True,
                     backpad = 0):
    # Separate data into timeseries and voxel data 
    run_timeseries = np.asarray(timeseries_data.filter(range(350)))
    print(run_timeseries.shape)
    voxel_data = timeseries_data.filter(['roi_int_labels', 'roi_labels', 'x', 'y', 'eccen', 'angle', 'sigma', 'vexpl'])

    if mean_subtract:
        voxel_mean = np.expand_dims(run_timeseries[:, :].mean(axis = 1), 1)
        run_timeseries = run_timeseries - voxel_mean
        

    # Define bins
    dist_bins = np.arange(-180, 220, binsize) - binsize/2
    center_bin = lambda x: (x.left.astype(float) + x.right.astype(float))/2

    Data = []
    for i, trial in design.iterrows():
        start = int(trial.cue_onset - backpad) 
        stop = start + window_size
        
        cut = run_timeseries[:, start:stop]

        # join with voxel_data 
        v_data = voxel_data.copy()
        for i, t in enumerate(range(-backpad, window_size-backpad)):
            v_data.insert(len(v_data.columns), column = t, value = cut[:, i])

        # Insert columns for angular distance and bins
        stim_angle = angles[trial.cond]

        theta = np.asarray([convert_angle(a) for a in np.degrees(v_data.angle)])
        target_dist = fix_deg(theta - stim_angle)

        v_data.insert(0, 'trialnum', trial.trialNum)
        v_data.insert(1, 'task', trial.block)
        v_data.insert(2, 'pref_angle', theta)
        v_data.insert(3, 'stim_angle', stim_angle)
        v_data.insert(4, 'target_dist', target_dist)

       
        bins = pd.cut(v_data['target_dist'], bins=dist_bins)
        bins = bins.apply(center_bin).astype(float)
        bins[bins == -180.0] = 180.0
        v_data.insert(8, 'ang_dist_bin', bins)

        Data.append(v_data)

    Data = pd.concat(Data)

    return Data

# %%
parser = argparse.ArgumentParser()

parser.add_argument("--subject", required = True, type=int)
parser.add_argument("--model", required = True, type=str)
parser.add_argument("--model_component", required = True, 
                    choices = ['all', 'p-ltm-wm', 'wmtarget', 'cue', 'sacc'])
parser.add_argument("--derivatives_dir", required = True)
parser.add_argument("--behavior_dir", required = True)
parser.add_argument("--binsize", required = False, default = 10)
parser.add_argument("--backpad", required = False, default = 0, type = int)
parser.add_argument("--window_size", required = False, default = 15, type = int)



args = parser.parse_args()

subj = args.subject
model = args.model
model_component = args.model_component
deriv_dir = os.path.expanduser(args.derivatives_dir)
behav_dir = os.path.expanduser(args.behavior_dir)
binsize = args.binsize
backpad = args.backpad
window_size = args.window_size

designs_dir = os.path.join(deriv_dir, 'design_matrices')
print("Building timeseries for: ", subj, model, model_component)


fname = os.path.join(deriv_dir, "GLMdenoise/%s/sub-wlsubj%03d/modelmd_2.txt" % (model, subj))
glm_fits = np.asarray(pd.read_csv(fname, sep = '\t', header = None))

fname = os.path.join(deriv_dir, "GLMdenoise/%s/sub-wlsubj%03d/modelmd_1.txt" % (model, subj))
hrf = np.asarray(pd.read_csv(fname, sep = '\t', header = None))

# Save as model predicted timeseries
save_dir = os.path.join(deriv_dir, "predicted_timeseries/%s/sub-wlsubj%03d/" % (model, subj))
if not os.path.exists(save_dir): os.makedirs(save_dir)

prf_dir = os.path.join(deriv_dir, "roi_labels")
pos_dir = os.path.join(behav_dir, 'positions')

target_ecc = 7

trialdata_dir = os.path.join(behav_dir, "results/sub-wlsubj%03d/*trialdata*" % subj)
if subj == 139:
    trialdata_dir = os.path.join(behav_dir, "results/sub-wlsubj%03d/*dateplaceholder_trialdata*" % subj)

trialdata = []
for trialdata_fname in glob.glob(trialdata_dir):
    df = pd.read_csv(trialdata_fname, sep = '\t', index_col = 0)
    df['run'] = int(trialdata_fname.split('_')[1].split("-")[1])
    trialdata.append(df)

trialdata = pd.concat(trialdata)
trialdata = trialdata.query("run > 0")

# Compute angular distance for each condition
pos_filepath = os.path.join(pos_dir, "sub-wlsubj%03d_16pos.tsv" % subj)
positions = pd.read_csv(pos_filepath, sep = "\t", index_col = 0)

angles = positions.degrees.values
tasks = positions.task.values
conds = positions.cond.values

# %%
# For each run

for run in tqdm.tqdm(range(12)):
    #   Read in design matrix 
    design_fname = os.path.join(designs_dir, "%s/sub-wlsubj%03d/sub-wlsubj%03d_run-%02d.tsv" % 
                                (model, subj, subj, run))
    design = np.asarray(pd.read_csv(design_fname, sep = '\t', index_col = 0, header = None))

    match model_component:
        case 'all':
            #   convolve design matrix with HRF, removing the tail end of the convolution to preserve shape
            X = convolve2d(design, hrf)[:design.shape[0], :]
            
            #   Multiply by glm estimates 
            Y_hat = glm_fits @ X.T
    
        case 'p-ltm-wm':
            # model timeseries for just task conditions
            design = design[:, :48]
            glm_fits_component = glm_fits[:, :48]

            #   convolve design matrix with HRF, removing the tail end of the convolution to preserve shape
            X = convolve2d(design, hrf)[:design.shape[0], :]
            
            #   Multiply by glm estimates 
            Y_hat = glm_fits_component @ X.T
 
        case 'cue':
            # model timeseries for just cue condition
            design = np.expand_dims(design[:, 48], axis = 1)
            glm_fits_component = np.expand_dims(glm_fits[:, 48], axis = 1)

            #   convolve design matrix with HRF, removing the tail end of the convolution to preserve shape
            X = convolve2d(design, hrf)[:design.shape[0], :]
            
            #   Multiply by glm estimates 
            Y_hat = glm_fits_component @ X.T


        case 'sacc':
            # model timeseries for just sacc condition
            design = np.expand_dims(design[:, 49], axis = 1)
            glm_fits_component = np.expand_dims(glm_fits[:, 49], axis = 1)

            #   convolve design matrix with HRF, removing the tail end of the convolution to preserve shape
            X = convolve2d(design, hrf)[:design.shape[0], :]
            
            #   Multiply by glm estimates 
            Y_hat = glm_fits_component @ X.T

        # model timeseries for wmtarget
        case 'wmtarget':
            design = design[:, 50:]
            glm_fits_component = glm_fits[:, 50:]

            #   convolve design matrix with HRF, removing the tail end of the convolution to preserve shape
            X = convolve2d(design, hrf)[:design.shape[0], :]
            
            #   Multiply by glm estimates 
            Y_hat = glm_fits_component @ X.T

    print("Building timeseries dataframes...")

    prf_params = load_retinotopy(
        os.path.join(prf_dir, "sub-wlsubj%03d" % subj), 
        ['lh', 'rh'], 
        ['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl', 'ROIs_V1-LO1'], 
        altdir = os.path.join(prf_dir, 'sub-wlsubj%03d' % subj))

    # Merge prf params and GLMdenoised timeseries
    data = combined_df(prf_params, Y_hat)
    data = data.rename(columns={"ROIs_V1-LO1": 'roi_int_labels'})

    data['roi_int_labels'] = [int(i) for i in data['roi_int_labels'].values]

    # Filter for only vertices that are labeled and fall within one sigma of target eccen, 
    # and filter for voxels with vexpl > 0.1
    data = data.query("roi_int_labels != 0 & vexpl >= 0.1")
    data = data[np.abs(target_ecc - data.eccen) <= data.sigma] 
    data['wlsubj'] = subj

    # Add a new column to map roi_int_labels to their respective roi map names
    data['roi_labels'] = map_roi_labels(data['roi_int_labels'].values)

    # Split data into the timeseries and voxel parameters
    run_ts = np.asarray(data.filter([str(i) for i in range(350)]))
    design = trialdata.query("run == @run+1")

    # Slice timeseries from run into trial timeseries, compute angular distances and assign bin values
    # The predicted timeseries contains many zeros. Mean subtracting pointless introduces zero divide errors
    trialwise_timeseries = slice_timeseries(data, design, angles, binsize=binsize, mean_subtract=True, 
                                            window_size=window_size, backpad=backpad)

    # Group by roi and ang_dist_bin
    TTA = trialwise_timeseries.groupby(['roi_labels', 'ang_dist_bin', 'task'], as_index = False).mean()


    if not os.path.exists(os.path.join(deriv_dir, 'dataframes/%s/sub-wlsubj%03d/timeseries/' % (model, subj))):
        os.makedirs(os.path.join(deriv_dir, 'dataframes/%s/sub-wlsubj%03d/timeseries/' % (model, subj)))

    tts_fname = os.path.join(deriv_dir, 'dataframes/%s/sub-wlsubj%03d/timeseries/pred_%s_trialwise_timeseries_run-%02d.tsv' % (model, subj, model_component, run+1))
    tta_fname = os.path.join(deriv_dir, 'dataframes/%s/sub-wlsubj%03d/timeseries/pred_%s_TTA_run-%02d.tsv' % (model, subj, model_component, run+1))

    trialwise_timeseries.to_csv(tts_fname, sep = '\t')
    print(tts_fname)
    TTA.to_csv(tta_fname, sep = '\t')
    print(tta_fname)

# %%




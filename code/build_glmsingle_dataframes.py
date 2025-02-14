# %%
import numpy as np
import os
import pandas as pd
import itertools
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.special import iv
import scipy.signal as sp
import matplotlib.pyplot as plt
from glob import glob
import glob
from interstellar_subjects import wlsubjects
import os.path as op
import nibabel as nib
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

parser = argparse.ArgumentParser()

parser.add_argument("--subject", required = True, type = int)
parser.add_argument("--derivatives_dir", required = True, type = str)
parser.add_argument("--old_glm", required = False, type = bool, default = False)
parser.add_argument("--target_ecc", required = False, type = float, default = 7)
parser.add_argument("--sigma_multiplier", required = False, type = float, default = 1)
parser.add_argument("--suffix", required = False, type = str, default = "")

args = parser.parse_args()

suffix = args.suffix

print(args)


# %%
def load_retinotopy(subj_prf_directory, load_hemis, ret_params, altdir = ''):
    ret = {p:{'lh':[], 'rh':[]} for p in ret_params}

    for hemi, param in itertools.product(load_hemis, ret_params):
        rfile = glob.glob(op.join(subj_prf_directory, '%s*%s.mgz') % (hemi, param))
        print(rfile)
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
    print("GLM", glm_fits.shape)
    prfs = {}

    for param in ret_params:
        try:
            prfs[param] = prf_params[param][hemi][0]
            print(param, prfs[param].shape)
        except:
            print("%s freesurfer parameter file not found..." % param)

    
    prfs = pd.DataFrame(prfs)
    assert prfs.shape[0] == glm_fits.shape[0], "ERROR: Vertex count not equal, check data..."

    DF = pd.DataFrame(prfs).join(glm_fits)

    return DF


def fix_deg(x):
    x = x - np.floor(x / 360 + 0.5) * 360
    
    return x


def convert_angle(x):
    if x < 0: x += 360
    
    return x



def norm_group(df, yvar = 'beta', xvar = 'ang_dist_bin', group_cols = []):
    # Take mean of obs w/in each data group and distance bin
    group_cols = ['wlsubj', 'task'] + group_cols

    
    data = df.groupby(group_cols + [xvar]).mean()[[yvar, 'vexpl']].reset_index()

    # Divide each subj response by norm
    norm_data = []
    for (cols, g) in data.groupby(group_cols):
        sd = g.copy()
        sd.loc[:, 'norm'] = np.linalg.norm(sd[yvar])
        sd.loc[:, yvar+'_norm'] = sd[yvar] / sd['norm']
        sd.loc[:, 'vexpl'] = sd['vexpl']

        norm_data.append(sd)

    norm_data = pd.concat(norm_data)

    # Average across subjects and multiply by average norm to get units back
    norm_data = norm_data.groupby(group_cols[1:] + [xvar]).mean().reset_index()
    norm_data[yvar+'_adj'] = norm_data[yvar+'_norm']*norm_data['norm']

    
    return norm_data


def norm_group_perc(df, yvar = 'beta', xvar = 'ang_dist_bin'):
    # Take mean of obs w/in each data group and distance bin
    group_cols = ['wlsubj', 'task', 'roi_labels']

    
    data = df.groupby(group_cols + [xvar]).mean()[[yvar, 'vexpl']].reset_index()

    subjects = data.wlsubj.unique()
    rois = data.roi_labels.unique()

    # Compute perc norm for each group
    perc_norm_dict = {}
    for subj, roi in itertools.product(subjects, rois):
        p_d = data.query("wlsubj == @subj & roi_labels == @roi & task == 'perception'")
        perc_norm_dict["%s_%s" % (str(subj), roi)] = np.linalg.norm(p_d[yvar])


    # Divide each subj response by norm
    norm_data = []
    for (cols, g) in data.groupby(group_cols):
        sd = g.copy()
        sd.loc[:, 'norm'] = np.linalg.norm(sd[yvar])
        sd.loc[:, yvar+'_norm'] = sd[yvar] / sd['norm']
        sd.loc[:, 'vexpl'] = sd['vexpl']

        subj = cols[0]
        roi = cols[2]

        sd.loc[:, 'norm_perc'] = perc_norm_dict["%s_%s" % (str(subj), roi)]
        sd.loc[:, yvar+'_norm_perc'] = sd[yvar] / sd['norm_perc']

        norm_data.append(sd)

    norm_data = pd.concat(norm_data)

    # Average across subjects and multiply by average norm to get units back
    norm_data = norm_data.groupby(group_cols[1:] + [xvar]).mean().reset_index()
    norm_data[yvar+'_adj'] = norm_data[yvar+'_norm']*norm_data['norm']
    norm_data[yvar+'_adj_perc'] = norm_data[yvar+'_norm_perc']*norm_data['norm_perc']

    return norm_data


def map_roi_labels(labels, roi_label_dict = roi_label_dict):
    if isinstance(labels[0], str):
        new_labels = [roi_label_dict[s] for s in labels]

    else:
        new_labels = [list(roi_label_dict.keys())[list(roi_label_dict.values()).index(s)] for s in labels]

    return new_labels


def load_glmsingle(filename, params = ''):
    g = np.load(filename, allow_pickle = True).item()
    
    if params == 'all':
        gparams = dict()
        for p, values in g.items():
            gparams[p] = g[p]
            
    elif params:
        gparams = dict()
        for p in params:
            gparams[p] = g[p]          
    else:
        gparams = pd.DataFrame(g['betasmd'].squeeze())

    return gparams



# %%
subj = args.subject
deriv_dir = os.path.expanduser(args.derivatives_dir)

prf_dir = os.path.join(deriv_dir, "roi_labels/sub-wlsubj%03d" % subj)
if args.old_glm:
    glm_dir = os.path.expanduser(os.path.join("~/mnt/winawer/Projects/Interstellar/analyses/GLMsingle/sub-wlsubj%03d" % subj))
else:
    glm_dir = os.path.join(deriv_dir, "GLMsingle/sub-wlsubj%03d" % subj)
    
pos_dir = os.path.join(deriv_dir, "behav/positions/")
eye_dir = os.path.join(deriv_dir, "behav/eyedata/")
expdesign_dir = os.path.join(deriv_dir, "behav/exp_design/sub-wlsubj%03d" % subj)

conds_file = 'sub-wlsubj%03d_conds.tsv' % subj

target_ecc = args.target_ecc
sigma_multiplier = args.sigma_multiplier

binsize = 20
dist_bins = np.arange(-180, 220, binsize) - binsize/2
center_bin = lambda x: (x.left.astype(float) + x.right.astype(float))/2


# %%
# Load experiment design data
pos_fname = os.path.join(pos_dir, "sub-wlsubj%03d_16pos.tsv" % subj)
positions = pd.read_csv(pos_fname, sep = '\t', index_col = 0)

angles = positions.degrees.values
tasks = positions.task.values
conds = positions.cond.values

conds_fname = os.path.join(expdesign_dir, 'sub-wlsubj%03d_conds.tsv' % subj)
conditions = pd.read_csv(conds_fname, sep = '\t', index_col = 0).conds.values

exclude_trials = wlsubjects['wlsubj%03d' % subj]['exclude_trials']
if exclude_trials:
    conditions = np.asarray([c for i, c in enumerate(conditions) if i not in exclude_trials])

# load saccades data and prep it for merging
saccades_fname = os.path.join(eye_dir, "sub-wlsubj%03d_saccades.tsv" % subj)
saccades = pd.read_csv(saccades_fname, sep = '\t').query('sacc_label == "response"')

saccades['event_id'] = (saccades.run - 1) * 16 + saccades.trial_id
saccades = saccades.filter(['event_id', 'sacc_theta', 'ang_dist', 'sac_ecc'])
saccades = saccades.rename(columns = {'sacc_theta': 'sacc_ang_rads', 'ang_dist': 'sacc_target_dist', 'sac_ecc':'sacc_ecc'})
saccades['sacc_ang_degs'] = np.degrees(saccades['sacc_ang_rads'])


# load retinotopy data
prf_params = load_retinotopy(
    prf_dir, 
    ['lh', 'rh'], 
    ['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl', 'ROIs_V1-LO1'], 
    altdir = os.path.join(prf_dir, 'sub-wlsubj%03d' % subj))

# Load glm fits
glm_fname = os.path.join(glm_dir, "TYPED_FITHRF_GLMDENOISE_RR.npy")
glm = load_glmsingle(glm_fname)



# %%
data = combined_df(prf_params, glm)

# Clean up roi label column
data = data.rename(columns={"ROIs_V1-LO1": 'roi_int_labels'})
    

data['roi_int_labels'] = [int(i) for i in data['roi_int_labels'].values]


# Filter for only vertices that are labeled and fall within one sigma of target eccen, 
# and filter for voxels with vexpl > 0.1
data = data.query("roi_int_labels != 0 & vexpl >= 0.1")
data = data[np.abs(target_ecc - data.eccen) <= (data.sigma * sigma_multiplier)] 
data['wlsubj'] = subj
print(data.roi_int_labels.unique())
# Add a new column to map roi_int_labels to their respective roi map names
data['roi_labels'] = map_roi_labels(data['roi_int_labels'].values)

# Relabel condition columns appropriately
numeric_columns = [str(c).isnumeric() for c in data.columns.values]
column_rename_dict = {old: "event_%02d" % int(old) for old in data.columns[numeric_columns]}
data = data.rename(columns = column_rename_dict)

# %%
df = []
for event_id, cond_num in enumerate(conditions):
    d = data.filter(['roi_int_labels', 'roi_labels', 'x', 'y', 'eccen', 'angle', 'sigma', 'vexpl', 'R2'])
    
    d.insert(0, 'wlsubj', subj)
    d.insert(1, 'task', tasks[cond_num])
    d.insert(2, 'cond', cond_num)
    d.insert(3, 'event_id', event_id)

    d.insert(len(d.columns), 'beta', data["event_%02d" % event_id])

    df.append(d)
    
df = pd.concat(df)
df = df.merge(saccades, on = "event_id")

stim_angle = angles[df.cond.values.astype(int)]

# convert pref angles to degrees. Algin to target/saccades
theta = np.asarray([convert_angle(a) for a in np.degrees(df.angle)])
target_dist = fix_deg(theta - stim_angle)
sacc_dist = fix_deg(theta - df.sacc_ang_degs)

df.insert(4, 'pref_angle', theta)
df.insert(5, 'stim_angle', stim_angle)
df.insert(6, 'target_dist', target_dist)
df.insert(7, 'sacc_dist', sacc_dist)

# Bin these distances
bins = pd.cut(df['target_dist'], bins=dist_bins)
bins = bins.apply(center_bin).astype(float)
bins[bins == -180.0] = 180.0
df.insert(8, 'ang_dist_bin', bins)

# Bin these distances
bins = pd.cut(df['sacc_dist'], bins=dist_bins)
bins = bins.apply(center_bin).astype(float)
bins[bins == -180.0] = 180.0
df.insert(8, 'sacc_ang_dist_bin', bins)


# Normalize data
# Aligned to targets
norm_data_target = norm_group_perc(df, xvar = 'ang_dist_bin')

# Aligned to saccades
# Before normalizing  saccades, filter out trials where saccade fell outside of 12 degrees eccentricity 
# — these saccades fell outside the display screen.
norm_data_saccade = norm_group_perc(df.query("sacc_ecc <= 12"), xvar = 'sacc_ang_dist_bin')

# Sacc split groups
if suffix == 'saccsplit':
    ds = []
    for task in df.task.unique():
        d = df.query("task == @task & sacc_ecc <= 12")

        tertile = pd.qcut(d.sacc_target_dist, 3, labels = ['counter', 'center', 'clock'])
        d.insert(len(d.columns), 'tertile', tertile)
        ds.append(d)

    ds = pd.concat(ds)
    df = df.merge(ds, how = 'left')

# Distance to nearest target

target_dists_to_nearest_target = []
for task in df.task.unique():
    d = df.query("task == @task")
    A = np.zeros([16, 16])

    for i, j in itertools.product(range(16), range(16)):
        A[i, j] = np.abs(fix_deg(d.stim_angle.unique()[i] - d.stim_angle.unique()[j]))
        if i == j: A[i, j] = np.inf

    dist_nearest = np.min(A, axis = 1)
    nearest = d.stim_angle.unique()[np.argmin(A, axis = 1)]

    D = pd.DataFrame({"stim_angle": d.stim_angle.unique(), "nearest_dist": dist_nearest, "task": task})
    target_dists_to_nearest_target.append(D)    

target_dists_to_nearest_target = pd.concat(target_dists_to_nearest_target).reset_index(drop = True)
df = df.merge(target_dists_to_nearest_target, on=['stim_angle', 'task'])

if args.old_glm:    
    dataframe_dir = os.path.join(deriv_dir, 'dataframes/old_glmsingle/sub-wlsubj%03d/' % subj)
else:
    dataframe_dir = os.path.join(deriv_dir, 'dataframes/glmsingle/sub-wlsubj%03d/' % subj)

if not os.path.exists(dataframe_dir): os.makedirs(dataframe_dir)

if not suffix:
    df.to_csv(os.path.join(dataframe_dir, "vertex+dist_glmsingle+prf+roi.tsv"), sep = '\t')
    norm_data_target.to_csv(os.path.join(dataframe_dir, "target_dist_normedbetas.tsv"), sep = '\t')
    norm_data_saccade.to_csv(os.path.join(dataframe_dir, "saccade_dist_normedbetas.tsv"), sep = '\t')

else:
    df.to_csv(os.path.join(dataframe_dir, "vertex+dist_glmsingle+prf+roi_%s.tsv" % suffix), sep = '\t')
    norm_data_target.to_csv(os.path.join(dataframe_dir, "target_dist_normedbetas_%s.tsv" % suffix), sep = '\t')
    norm_data_saccade.to_csv(os.path.join(dataframe_dir, "saccade_dist_normedbetas_%s.tsv" % suffix), sep = '\t')

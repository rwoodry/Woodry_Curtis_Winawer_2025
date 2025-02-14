# %% 
import numpy as np
import pandas as pd
import os
import glob
import sys
from scipy.io import loadmat
import re
import itertools
import os.path as op
import nibabel as nib
import argparse

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

    DF = pd.DataFrame(glm_fits).join(prfs)

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


# %%

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subject", required = True, type = int)
    parser.add_argument("--derivatives_dir", required = True, type = str)
    parser.add_argument("--model", required = True, type = str)

    args = parser.parse_args()


    deriv_dir = os.path.expanduser(args.derivatives_dir)
    model = args.model
    subj = args.subject
    
    pos_dir = os.path.join(deriv_dir, 'behav/positions/')
    glm_output_dir = os.path.join(deriv_dir, "GLMdenoise/%s/" % model)
    prf_dir = os.path.join(deriv_dir, "roi_labels/")


    target_ecc = 7

    binsize = 20
    dist_bins = np.arange(-180, 220, binsize) - binsize/2
    center_bin = lambda x: (x.left.astype(float) + x.right.astype(float))/2

    roi_label_dict = {
        'V1': 1,
        'V2': 2,
        'V3': 3,
        'V4': 4,
        'LO1': 7,
        'V3ab': 5,
        'IPS0': 6
    }

    def map_roi_labels(labels, roi_label_dict = roi_label_dict):
        if isinstance(labels[0], str):
            new_labels = [roi_label_dict[s] for s in labels]

        else:
            new_labels = [list(roi_label_dict.keys())[list(roi_label_dict.values()).index(s)] for s in labels]

        return new_labels
            
    # reg_fname = os.path.join(glm_output_dir, "sub-wlsubj%03d/results_modelmd_sub-wlsubj%03d%s.mat" % (subj, subj, suffix))
    glmfits_fname = os.path.join(glm_output_dir, "sub-wlsubj%03d/modelmd_2.txt" % subj)
    # glm_fits = loadmat(glmfits_fname)['file_to_save'][0][1]
    glm_fits = pd.read_csv(glmfits_fname, sep = '\t', header = None)
    print(glmfits_fname, glm_fits.shape)

    # Had to use mri_surf2surf to update roi_labels based on updated freesurfer subject files in $SUBJECTS_DIR
    prf_params = load_retinotopy(
        os.path.join(prf_dir, "sub-wlsubj%03d" % subj), 
        ['lh', 'rh'], 
        ['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl', 'ROIs_V1-LO1'], 
        altdir = os.path.join(prf_dir, 'sub-wlsubj%03d' % subj))

        # Merge prf params and GLM fits
    data = combined_df(prf_params, glm_fits)

        # Clean up roi label column
    data = data.rename(columns={"ROIs_V1-LO1": 'roi_int_labels'})
            

    data['roi_int_labels'] = [int(i) for i in data['roi_int_labels'].values]
    

    # Filter for only vertices that are labeled and fall within one sigma of target eccen, 
    # and filter for voxels with vexpl > 0.1
    data = data.query("roi_int_labels != 0 & vexpl >= 0.1")
    data = data[np.abs(target_ecc - data.eccen) <= data.sigma] 
    data['wlsubj'] = subj
    print(data.roi_int_labels.unique())
    # Add a new column to map roi_int_labels to their respective roi map names
    data['roi_labels'] = map_roi_labels(data['roi_int_labels'].values)

    # Relabel condition columns appropriately
    numeric_columns = [str(c).isnumeric() for c in data.columns.values]
    column_rename_dict = {old: "cond_%02d" % int(old) for old in data.columns[numeric_columns]}
    data = data.rename(columns = column_rename_dict)
    
    # Load design dir
    design_dir = os.path.expanduser(os.path.join(deriv_dir, "behav/exp_design/sub-wlsubj%03d" % subj, "*trialdesign.tsv"))
    Designs = []
    for design_fname in glob.glob(design_dir):
        Designs.append(pd.read_csv(design_fname, sep = '\t', index_col = 0))

    Designs = pd.concat(Designs)

    # Compute angular distance for each condition
    pos_filepath = os.path.join(pos_dir, "sub-wlsubj%03d_16pos.tsv" % subj)
    positions = pd.read_csv(pos_filepath, sep = "\t", index_col = 0)

    angles = positions.degrees.values
    tasks = positions.task.values
    conds = positions.cond.values

    if model == 'p-ltm-wm-cue-sacc-wmtarget':
        wm_target_conds = list(range(50, 66))
        
        conds = np.concatenate((conds, wm_target_conds), axis = None)
        angles = np.concatenate((angles, [0, 0], angles[-16:]), axis = None)
        tasks = np.concatenate((tasks, ['cue', 'sacc'], ['wm_target'] * 16), axis = None)
        
    
    df = []
    for cond_num in conds:
        d = data.filter(['roi_int_labels', 'roi_labels', 'x', 'y', 'eccen', 'angle', 'sigma', 'vexpl'])
        
        d.insert(0, 'wlsubj', subj)
        d.insert(1, 'task', tasks[cond_num])
        d.insert(2, 'cond', cond_num)
        d.insert(len(d.columns), 'beta', data["cond_%02d" % cond_num])

        df.append(d)
        
    df = pd.concat(df)

    stim_angle = angles[df.cond.values.astype(int)]

    theta = np.asarray([convert_angle(a) for a in np.degrees(df.angle)])
    target_dist = fix_deg(theta - stim_angle)

    df.insert(4, 'pref_angle', theta)
    df.insert(5, 'stim_angle', stim_angle)
    df.insert(6, 'target_dist', target_dist)

    # Bin these distances
    bins = pd.cut(df['target_dist'], bins=dist_bins)
    bins = bins.apply(center_bin).astype(float)
    bins[bins == -180.0] = 180.0
    df.insert(8, 'ang_dist_bin', bins)

    # Normalize data
    norm_data = norm_group(df, xvar = 'ang_dist_bin', group_cols=['roi_labels'])

    # Save to tsv file in dataframes
    dataframe_dir = os.path.join(deriv_dir, 'dataframes/%s/sub-wlsubj%03d/' % (model, subj))
    if not os.path.exists(dataframe_dir): os.makedirs(dataframe_dir)
    
    data.to_csv(os.path.join(dataframe_dir, "vertex_glmdenoise+prf+roi.tsv"), sep = '\t')
    Designs.to_csv(os.path.join(dataframe_dir, "trialdesigns.tsv"), sep = '\t')
    positions.to_csv(os.path.join(dataframe_dir, "cond_positions.tsv"), sep = '\t')
    df.to_csv(os.path.join(dataframe_dir, "vertex+dist_glmdenoise+prf+roi.tsv"), sep = '\t')
    norm_data.to_csv(os.path.join(dataframe_dir, "dist_normedbetas.tsv"), sep = '\t')


if __name__ == "__main__":
    main()


    # %%

    # %%

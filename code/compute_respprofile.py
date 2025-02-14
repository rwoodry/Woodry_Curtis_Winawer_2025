import pandas as pd
import numpy as np
import os
import sys
import argparse
from scipy.optimize import curve_fit
import scipy.signal as sp
from scipy.special import iv
import itertools



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


def norm_group_perc(df, yvar = 'beta', xvar = 'ang_dist_bin', group_by = ""):
    # Take mean of obs w/in each data group and distance bin
    group_cols = ['wlsubj', 'task', 'roi_labels']

    if group_by: group_cols.append(group_by)

    
    data = df.groupby(group_cols + [xvar]).mean(numeric_only = True)[[yvar, 'vexpl']].reset_index()

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


def list_of_ints(arg):
    return [int(a) for a in arg.split(" ")]


def pair_of_strs(arg):
    return arg.split(" ")


def vonmises(theta, loc, kappa, scale):
    p = scale * np.exp(kappa*np.cos(theta-loc))/(2*np.pi*iv(0,kappa))
    return p


def diff_vonmises(theta, loc, kappa1, scale1, kappa2, scale2):
    p1 = vonmises(theta, loc, kappa1, scale1)
    p2 = vonmises(theta, loc, kappa2, scale2)
    return (p1 - p2) 


def fit_diff_vonmises(data, yvar, xvar = 'ang_dist_bin', group_cols=[], drop_cols=[]):
    # convert dist bins to radians
    data[xvar+'_rad'] = data[xvar].apply(np.deg2rad)
    
    # Highly sampled x range in radians
    x = np.deg2rad(np.arange(-180, 180, 1))
    
    params = []
    assumed_cols = list(filter(lambda x: x not in drop_cols, ['roi_labels', 'task']))
    sigma = None
    
    for cols, g in data.groupby(assumed_cols + group_cols):
        #try:
        bounds = [[-np.pi, 0, 0, 0, 0], [np.pi, np.inf, np.inf, np.inf, np.inf]]
        xbin = g[xvar+'_rad'].values
        ybin = g[yvar].values
            
        
        p_opt, p_cov = curve_fit(diff_vonmises, xbin, ybin, bounds=bounds, maxfev=100000, sigma = sigma)

        y_hat = diff_vonmises(x, *p_opt)

        loc, kappa1, scale1, kappa2, scale2 = p_opt

        #width = fwhm(x, y_hat)
        fwhm, _, _, _ = sp.peak_widths(
                     y_hat, np.where(y_hat == y_hat.max())[0])
        # Return relevant params
        p = dict(func='diff_vonmises',
                 loc=loc,
                 loc_deg=np.rad2deg(loc),
                 kappa1=kappa1, 
                 scale1=scale1,
                 kappa2=kappa2,
                 scale2=scale2,
                 maxr=max(y_hat),
                 minr=min(y_hat),
                 amp=max(y_hat)-min(y_hat),
                 fwhm = fwhm[0]
                )
        
        group_df = g.reset_index()[assumed_cols + group_cols].iloc[:1]
        p = pd.DataFrame(p, index=[0])
        p = pd.concat([group_df, p], axis = 1)
        params.append(p)
        
    params = pd.concat(params, sort = False).reset_index(drop=True)
    
    return params


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True, 
                        choices = ['glmsingle', 'old_glmsingle', 'p-ltm-wm', 'p-ltm-wm-cue-sacc', 'p-ltm-wm-cue-sacc-wmtarget'])
    parser.add_argument('--subjects', type = list_of_ints, required = True, help = 'string of subject id nums separated by a space, i.e. "114 115 116"')
    parser.add_argument('--xvar', type = str, required = False, default = 'ang_dist_bin')
    parser.add_argument('--group_by', type = str, required = False, default = '')
    parser.add_argument('--derivatives_dir', type = str, required = True)
    parser.add_argument('--suffix', type = str, required = False, default = "")
    parser.add_argument('--target_file', type = str, required = False, default = "vertex+dist_glmsingle+prf+roi.tsv")
    parser.add_argument('--bootstrap', type=pair_of_strs, required = False, default = "", help = "string of two values, the first being the bootstrap directory, the second being the bootstrap number")

    args = parser.parse_args()


    print('Subject id nums:')
    [print("\t- ", s) for s in args.subjects]

    xvar = args.xvar
    model = args.model
    deriv_dir = os.path.expanduser(args.derivatives_dir)
    target_file = args.target_file
    suffix = args.suffix
    group_by = args.group_by

    print("Model:", model)
    dataframes_dir = os.path.join(deriv_dir, "dataframes/%s" % model)

    data = []
    for subj in args.subjects:
        fname = os.path.join(dataframes_dir, "sub-wlsubj%03d" % subj, target_file)

        df = pd.read_csv(fname, sep = '\t', index_col = 0)
        data.append(df)

    data = pd.concat(data)

    if group_by == "nearest_dist_bin":
        data['nearest_dist_bin'] = pd.qcut(data.nearest_dist, 3, labels = ['near', 'mid', 'far']).values

    print("Normalizing data...")
    norm_data = norm_group_perc(data, xvar = xvar, group_by = group_by)

    if not args.bootstrap[0]: 
        fname = os.path.join(dataframes_dir, "allsubj_dist_normedbetas_%s.tsv" % xvar)
        if suffix: fname = os.path.join(dataframes_dir, "allsubj_dist_normedbetas_%s_%s.tsv" % (xvar, suffix))

    else:
        bootstrap_dir = args.bootstrap[0]
        boot_num = args.bootstrap[1]

        fname = os.path.join(bootstrap_dir, model, "boot_%d_allsubj_dist_normedbetas.tsv" % boot_num)
        
    norm_data.insert(0, 'model', model)

    print("Saving norm data...")
    norm_data.to_csv(fname, sep = '\t')
    
    # Fit diff_vonmises
    print("Estimating Von Mises fits...")
    vonmises_fits = fit_diff_vonmises(norm_data, 'beta_adj_perc', xvar=xvar)
    if group_by:
            vonmises_fits = fit_diff_vonmises(norm_data, 'beta_adj_perc', xvar=xvar, group_cols = [group_by])

    vonmises_fits.insert(0, 'model', model)

    vm_fits_fname = os.path.join(dataframes_dir, "allsubj_vmfits_%s.tsv" % xvar)
    
    if suffix: vm_fits_fname = os.path.join(dataframes_dir, "allsubj_vmfits_%s_%s.tsv" % (xvar, suffix))

    print("Saving vmfits...")
    vonmises_fits.to_csv(vm_fits_fname, sep = '\t')


if __name__ == "__main__":
    main()
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import tqdm
import seaborn as sns
import itertools
from scipy.special import iv
from scipy.optimize import curve_fit
import scipy.signal as sp
import scipy
import warnings
from IPython.display import HTML
import matplotlib.animation
import random
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

design_dir = '~/mnt/winawer/Projects/Interstellar/task/Interstellar/main/design/'
fmriprep_dir = '~/mnt/winawer/Projects/Interstellar/derivatives/fmriprep'
output_dir = '~/mnt/winawer/Projects/Interstellar/analyses/voxels/sub-wlsubj%03d/'
glm_dir = '~/mnt/winawer/Projects/Interstellar/analyses/GLMsingle'
voxels_dir = '~/mnt/winawer/Projects/Interstellar/analyses/voxels/sub-wlsubj%03d'
results_dir = '~/mnt/winawer/Projects/Interstellar/task/Interstellar/main/results/'
training_dir = '~/mnt/winawer/Projects/Interstellar/task/Interstellar/training/results/'
pos_dir = '~/mnt/winawer/Projects/Interstellar/task/Interstellar/training/positions/'
voxel_dir = '~/mnt/winawer/Projects/Interstellar/analyses/voxels/'
df_dir = '~/mnt/winawer/Projects/Interstellar/analyses/code/dataframes'

design_dir = os.path.expanduser(design_dir)
glm_dir = os.path.expanduser(glm_dir)
output_dir = os.path.expanduser(output_dir)
voxels_dir = os.path.expanduser(voxels_dir)
fmriprep_dir = os.path.expanduser(fmriprep_dir)
training_dir = os.path.expanduser(training_dir)
pos_dir = os.path.expanduser(pos_dir)
df_dir = os.path.expanduser(df_dir)

task_colors = {
    'perception': 'teal',
    'wm': 'green',
    'ltm': 'orange'
}

def load_conds(pos_dir, wlsubj):
    pos_filepath = os.path.join(pos_dir, "sub-wlsubj%03d_16pos.tsv" % wlsubj)
    conds = pd.read_csv(pos_filepath, sep = "\t", index_col = 0)
    
    return conds


def vonmises(theta, loc, kappa, scale):
    p = scale * np.exp(kappa*np.cos(theta-loc))/(2*np.pi*iv(0,kappa))
    return p


def fwhm(X, Y):
    d = Y - (max(Y) / 2) 
    indexes = np.where(d > 0)[0] 
    return abs(X[indexes[-1]] - X[indexes[0]])


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    return(rho, phi)


def pol2cart(rho, phi):    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    return(x, y)



def fix_deg(x):
    x = x - np.floor(x / 360 + 0.5) * 360
    
    return x


def load_conds(pos_dir, wlsubj):
    pos_filepath = os.path.join(pos_dir, "sub-wlsubj%03d_16pos.tsv" % wlsubj)
    conds = pd.read_csv(pos_filepath, sep = "\t", index_col = 0)
    
    return conds


def convert_angle(x):
    if x < 0: x += 360
    
    return x


center_bin = lambda x: (x.left.astype(float) + x.right.astype(float))/2


def diff_vonmises(theta, loc, kappa1, scale1, kappa2, scale2):
    p1 = vonmises(theta, loc, kappa1, scale1)
    p2 = vonmises(theta, loc, kappa2, scale2)
    return (p1 - p2) 


def norm_group(voxels, yvar = 'beta', xvar = 'ang_dist_bin', group_cols = [], precision = False):
    # Take mean of obs w/in each data group and distance bin
    group_cols = ['subj', 'roi', 'task'] + group_cols
    data = voxels.groupby(group_cols + [xvar]).mean()[[yvar, 'vexpl']].reset_index()
    
    if precision:
        d = voxels.groupby(group_cols + [xvar])
        prec = []
        precvar = []
        for cols, g in d:
            pwm, pwv = precision_weighted_mean(g['beta'], g['vexpl'])
            prec.append(pwm)
            precvar.append(pwv)
        
        data['prec'] = prec
        data['precvar'] = precvar
    
    # Divide each subj response by norm
    norm_data = []
    for (cols, g) in data.groupby(group_cols):
        sd = g.copy()
        sd.loc[:, 'norm'] = np.linalg.norm(sd[yvar])
        sd.loc[:, yvar+'_norm'] = sd[yvar] / sd['norm']
        sd.loc[:, 'vexpl'] = sd['vexpl']
        
        if precision:
            sd.loc[:, 'precnorm'] = np.linalg.norm(sd['prec'])
            sd.loc[:, yvar + '_precnorm'] = sd['prec'] / sd['precnorm']
            sd.loc[:, 'precvar'] = sd['precvar']
        norm_data.append(sd)
    
    norm_data = pd.concat(norm_data)
    
    # Average across subjects and multiply by average norm to get units back
    norm_data = norm_data.groupby(group_cols[1:] + [xvar]).mean().reset_index()
    norm_data[yvar+'_adj'] = norm_data[yvar+'_norm']*norm_data['norm']
    if precision: norm_data[yvar+'_adj_prec'] = norm_data[yvar+'_precnorm']*norm_data['precnorm']

    
    return norm_data
    
    
def compute_TTA(wlsubj, timeseries, vox, design, angles, n_trials, task_list, 
                windowwidth = 15, backpad = 0, bin_size = 20, output_dir = ''):
    P = pd.DataFrame()
    dist_bins = np.arange(-180, 220, bin_size) - 10

    for j, run in enumerate(timeseries):
        task = task_list[j]
        starts = np.where(design[j] == 1)[0] - backpad
        ends = starts + windowwidth + backpad
        conds_inorder = np.where(design[j] == 1)[1]

        # Convert to PCT Signal Change
        # For each voxel, compute percent signal change
        voxel_mean = np.expand_dims(run[:, :].mean(axis = 1), 1)
        voxel_diff = run - voxel_mean
        pct_signal_change = voxel_diff / voxel_mean * 100


        for i in tqdm.tqdm(range(n_trials)):
            TS = pct_signal_change[:, starts[i]:ends[i]]
            # get event code based on i(trialnum) and j(run num)
            # Get trial's saccade dist from sacc-aligned.tsv based on event code

            stim_angle = angles[conds_inorder[i]]
            theta = np.asarray([convert_angle(a) for a in np.degrees(vox.angle)])
            target_dists = fix_deg(theta - stim_angle)

            TS = pd.concat([vox.reset_index(), pd.DataFrame(TS)], axis = 1)
            TS.insert(0, 'task', task)
            TS.insert(1, 'cond', conds_inorder[i])
            TS.insert(7, 'pref_angle', theta) 
            TS.insert(8, 'stim_angle', stim_angle)
            TS.insert(9, 'ang_dist', target_dists)

            # Bin these distances
            bins = pd.cut(TS['ang_dist'], bins=dist_bins)
            bins = bins.apply(center_bin).astype(float)
            bins[bins == -180.0] = 180.0
            TS.insert(11, 'ang_dist_bin', bins)

            P = P.append(TS)

    rename_cols = {}
    for i in range(windowwidth+backpad):
        rename_cols[i] = str(i)
    P = P.rename(columns = rename_cols)
    P.insert(0, 'subj', wlsubj)

    for  i in range(windowwidth+backpad):
        if not i:
            Pnorm = norm_group(P, yvar = str(i))
        if i:
            b = norm_group(P, yvar = str(i))
            Pnorm.insert(Pnorm.shape[1], str(i), b[str(i)])
            Pnorm.insert(Pnorm.shape[1], str(i)+'_adj', b[str(i)+'_adj'])
    if output_dir:  
        save_name_norm = "sub-wlsubj%03d_TTA_norm_b%02dw%02d_bs%02d.tsv" % (
            wlsubj, backpad, windowwidth, bin_size
        )
        save_name = "sub-wlsubj%03d_TTA_b%02dw%02d_bs%02d.tsv" % (
            wlsubj, backpad, windowwidth, bin_size
        )
        
        save_name = os.path.join(output_dir, save_name)
        save_name_norm = os.path.join(output_dir, save_name_norm)
        print("Saving to %s" % save_name)
        Pnorm.to_csv(save_name_norm, sep = '\t')
        P.to_csv(save_name, sep = "\t")
            
    return Pnorm


def plot_TTA_old(data, group_cols = ['roi', 'task'], adj = True, figsize=(16,8),
            windowwidth = 15, backpad = 0, save = '', titles = True, order = []):
    rois = list(data.roi.unique())
    rois.sort()
    
    grouped = data.groupby(group_cols)

    nrows = data.roi.nunique()
    ncols = data.task.nunique()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex = True, sharey=True)

    for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
        roi, task = key
        
        if order:
            col = order.index(task)
            row = rois.index(roi)
            ax = axes[row, col]

        data = grouped.get_group(key)
        if adj:
            data = data.filter(regex = 'adj')
        else:
            data = data.filter([col for col in data.columns if col.isnumeric()])

        p = ax.imshow(data, aspect = 'auto', interpolation = 'gaussian', cmap = 'twilight_shifted', 
                  vmax = 1,
                  vmin = -1,
                  extent = [0, windowwidth+backpad, -180, 180])

        title = "V%d %s" % (roi, task.title())
        if titles: ax.set_title(title)
        ax.set_yticks(np.arange(-180, 220, 90))
        ax.set_xticks(np.arange(0, windowwidth+backpad, 1))
        ax.set_xticklabels(np.arange(-backpad, windowwidth, 1))
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_yticks([-100, 0, 100])

    fig.colorbar(p, ax=ax)
        
    plt.tight_layout()
    if save:
        plt.savefig(save, facecolor = 'white')


def plot_TTA(data, fits, group_cols = ['roi_labels', 'task'], adj = True, figsize=(16,16),
            windowwidth = 15, backpad = 0, save = '', titles = False, order = [], center = True,
            cmap = "twilight_shifted", interpolation = 'gaussian', start = 0, pad = 5, 
            colors = {
                'perception': 'teal',
                'ltm':'gold',
                'wm':'green'
            }
            ):
    rois = list(data.roi.unique())
    rois.sort()
    
    grouped = data.groupby(group_cols)

    nrows = data.roi.nunique()
    ncols = data.task.nunique()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex = True, sharey=True)
    
    for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
        roi, task = key
        
        if order:
            col = order['tasks'].index(task)
            row = order['rois'].index(roi)
            ax = axes[row, col]

        data = grouped.get_group(key)
        if adj:
            data = data.filter(regex = 'adj')
        else:
            data = data.filter([col for col in data.columns if col.isnumeric()])

        ax.imshow(data, aspect = 'auto', interpolation = interpolation, cmap = cmap, 
                  vmax = 1,
                  vmin = -1,
                  extent = [0, windowwidth+backpad, -180, 180])

        # If fits, plot FWHM estimates
    
        params = fits.query("roi_labels==@roi & task == @task")
        f = params.fwhm.values
        #f = np.append(f, f[-1]) 
        t = params.tr.values
        #t = np.append(t, t[-1]+1)
        a = params.amp.values
        a = np.append(a, a[-1]+1)
        l = params.loc_deg.values
        
        if center: 
            l=0
            ax.plot([0, t[-1]+1], [0, 0], linestyle = ":", color = 'dimgray')
        else:
            l = np.append(l[3:], l[-1])

        f = np.append(f[start:], f[-1])
        t = np.append(t[start:], t[-1]+0.5)
    
        ax.plot(t+0.5, f/2 - l, color = "white", linestyle = ":", alpha = .75, linewidth = 2.5)
        ax.plot(t+0.5, -f/2 - l, color = "white", linestyle = ":", alpha = .75, linewidth = 2.5)
        if not center: ax.plot(t+0.5, -l, color = "dimgray", linestyle = ":")
        ax.plot([3.5/2, 3.5/2], [-180, 180], linewidth = 66, alpha = 0.5, color = "white")

        title = "%s %s" % (roi, task.title())

        # Axes parameters & aesthetics
        if titles: ax.set_title(title)
        ax.set_yticks(np.arange(-180, 220, 90))
        ax.set_xticks(np.arange(0, windowwidth+backpad, 1))
        ax.set_xticklabels(np.arange(-backpad, windowwidth, 1))
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xticks(np.asarray([0, 2, 4, 6, 8, 10, 12, 14]) + 0.5)
        ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12, 14], size = 14)
        ax.set_yticks([-100, 0, 100])
        ax.set_yticklabels([-100, 0, 100], size = 14)
        if (col == 0) & (row == 3): ax.set_ylabel("Polar Angle Distance from Target (in Degrees)", size = 18, color = 'dimgray')
        if (col == 0) & (row != 3): ax.set_ylabel(" ", size = 15)
        if (row == len(order['rois'])-1) & (col == 1): ax.set_xlabel("Time since Cue (in Seconds)", size = 18, color = 'dimgray')

        for spine in ax.spines.values():
            spine.set_edgecolor(colors[task])
            spine.set_linewidth(0.1)


    # for i, roi in enumerate(order['rois']):
    #     x = 0
    #     y = 1 - i/len(order['rois']) - 1/((len(order['rois'])*2))
    #     fig.text(x, y, '%s' % (roi), verticalalignment = 'center', horizontalalignment='center')

    # Figure Aesthetics
    for ax, col in zip(axes[0], order['tasks']):

        if col == 'perception': ax.set_title(col.title(), size = 18)
        if col == 'ltm': ax.set_title("Long-Term Memory", size = 18)
        if col == 'wm': ax.set_title("Working Memory", size = 18)


    for ax, row in zip(axes[:,0], order['rois']):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=18, ha='right', va='center')


    plt.tight_layout()
    if save:
        plt.savefig(save, transparent=True)

        
def load_TTA(wlsubj, tta_dir, suffix = ''):
    fname = "sub-wlsubj%03d_TTA_%s" % (wlsubj, suffix)
    tta_path = os.path.join(tta_dir, "sub-wlsubj%03d" % wlsubj, fname)
    
    if not suffix:
        print("Must provide TTA file suffix. \nSome possible suffixes for desired wlsubj:\n------------------------------")
        possible_filenames = glob.glob(os.path.join(tta_dir, "sub-wlsubj%03d" % wlsubj, "*TTA*.tsv"))
        [print(i.split("/")[-1].split("sub-wlsubj%03d_TTA_" % wlsubj)[-1]) for i in possible_filenames]
        
    else:
        tta = pd.read_csv(tta_path, sep="\t", index_col = 0)
    
        return tta
    
    
def TTA_allsubjects(wlsubjects, tta_dir, suffix, output_dir = '', drop_subj = False):
    tta = []
    
    print("Loading subjects...")
    for wlsubj in tqdm.tqdm(wlsubjects):
        fname = "sub-wlsubj%03d_TTA_%s" % (wlsubj, suffix)
        tta_path = os.path.join(tta_dir, "sub-wlsubj%03d" % wlsubj, fname)
        tta.append(pd.read_csv(tta_path, sep="\t", index_col = 0))
    
    TTA = pd.concat(tta, ignore_index=True, axis=0)
    
    if drop_subj:
        TTA.subj = 0
    
    
    n_TRs = sum([i.isnumeric() for i in TTA.columns])
    print(n_TRs)
    
    print("Normalising by group keys...")
    
    B = []
    for i in tqdm.tqdm(range(n_TRs)):
        if not i:
            TTA_n = norm_group(TTA, yvar = str(i))
        else:
            b = norm_group(TTA, yvar = str(i)).filter(regex='adj')
            TTA_n.insert(TTA_n.shape[1], str(i) + '_adj', b)
            # TTA_n.insert(TTA_n.shape[1], str(i), b[str(i)])
    
            
    if output_dir:  
        save_name_norm = "sub-all_TTA_norm_.tsv"
        save_name_norm = os.path.join(output_dir, save_name_norm)
        save_name = "sub-all_TTA.tsv"
        save_name = os.path.join(output_dir, save_name)
        
        TTA_n.to_csv(save_name_norm, sep = '\t')
        TTA.to_csv(save_name, sep = '\t')
            
    return TTA_n, TTA
    
    
def fit_diff_vonmises(data, yvar, xvar = 'ang_dist_bin', group_cols=[], drop_cols=[], precision = False):
    # convert dist bins to radians
    data[xvar+'_rad'] = data[xvar].apply(np.deg2rad)
    
    # Highly sampled x range in radians
    x = np.deg2rad(np.arange(-180, 180, 1))
    
    params = []
    assumed_cols = list(filter(lambda x: x not in drop_cols, ['roi', 'task']))
    
    
    for cols, g in data.groupby(assumed_cols + group_cols):
        #try:
        bounds = [[-np.pi, 0, 0, 0, 0], [np.pi, np.inf, np.inf, np.inf, np.inf]]
        xbin = g[xvar+'_rad'].values
        ybin = g[yvar].values
        if precision: 
            sigma = np.sqrt(g['precvar'].values)
        else:
            sigma = None
            
        
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
        # except:
        #     print("FAIL")
        #     p = dict()
        
        group_df = g.reset_index()[assumed_cols + group_cols].iloc[:1]
        p = pd.DataFrame(p, index=[0])
        p = pd.concat([group_df, p], axis = 1)
        params.append(p)
        
    params = pd.concat(params, sort = False).reset_index(drop=True)
    
    return params


def fidelity(r, theta, d = 0):
    '''
    where r is the value of the representation in arbitrary units 
    and θ is the polar angle in radians. d is size of bins in radians for correction factor.
    Default is 0, i.e. no correction.
    '''   
    # Offset representation such that min is 0
    r_min = r.min()
    r = r.copy() + r_min
    
    f = np.max(r) * circ_r(theta, r, d = d) * np.cos(circ_mean(theta, r))
    
    return f


def circ_r(alpha, w = np.asarray([]), d = 0, dim = 0):
    '''
    Computes mean resultant vector length for circular data.
    
    Input:
        - alpha: sample of angles in radians
        - w: (optional) number of incidences in case of binned angle data
        - d: (optional) spacing of bin centers for binned data, i supplied correction factor is used 
            to correct of bias in estimation of r, in radians
        - dim: (optional) compute along this dimension, default: 1st non-singular dimension
        
    Output:
        r: mean resultant length
        
    *** Code Adapted to Python from the Circular Statistics Toolbox for MAtlab, 
    by Philipp Berens, 2009
    '''
    
    if not np.any(w):
        w = np.ones(alpha.shape)
        
    else:
        if w.shape != alpha.shape:
            raise Exception("Alpha and W dimensions dot match! %d %d" % (alpha.shape, w.shape))
    
    # Compute weighted sum of cos and sin angles
    r = np.sum(w * np.exp(1j * alpha), dim)
    
    # obtain length
    r = np.abs(r) / np.sum(w, dim)
    
    # If binned data, apply bias correction
    if d:
        c = d/ 2 / np.sin(d / 2)
        r = c * r
        
    return r


def circ_mean(alpha, w = np.asarray([]), dim = 0, ci = False):
    '''
    Computes the mean direction for circular data.
    
    Input:
        - alpha: sample of angles in radians
        - w: (optional) weightings in case of binned angle data
        - dim: (optional) compute along this dimension. Default is 1st non-singular dimension
        
    Output:
        - mu: mean direction
        - ul: upper 95% confidence limit
        - ll: lower 95% confidence limit
        
    *** Code Adapted to Python from the Circular Statistics Toolbox for MAtlab, 
        by Philipp Berens, 2009
    '''
    
    if not np.any(w):
        w = np.ones(alpha.shape)
        
    
    else:
        if w.shape != alpha.shape:
            raise Exception('Alpha and W dims do not match! %d %d' 
                            % (alpha.shape, w.shape))
    
    # Compute weighted sum of cos and sin of angles
    r = np.sum(w * np.exp(1j * alpha), dim)
    
    # obtain mean
    mu = np.angle(r)
    
    return mu
    
    # if Confidence limits desired, output
    # add code here later


def get_angdist_bins(wlsubj, prevoxels, conds_by_trials, tasks, conds, angles, binsize = 20, query = ''):
    if query: prevoxels = prevoxels.query(query)

    voxels_by_stimangle = prevoxels.filter(['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl', 'roi',
           'surf_label'])

    voxel_dfs = {}

    for c in tqdm.tqdm(conds):
        idx = np.where(conds_by_trials == c)[0]

        v = prevoxels.iloc[:, idx]
        v = v.mean(axis = 1)

        voxels_by_stimangle.insert(int(c), str(c), v)


    for t in tqdm.tqdm(np.unique(tasks)):
        voxel_dfs[t] = prevoxels.filter(['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl',
           'surf_label', 'roi'])
        voxel_dfs[t]['task'] = t
        idx = np.where(tasks == t)[0]

        voxel_dfs[t] = voxel_dfs[t].merge(voxels_by_stimangle.iloc[:, idx], left_index = True, right_index = True)
        voxel_dfs[t] = voxel_dfs[t].melt(id_vars=['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl', 'surf_label', 'task', 'roi'], 
            var_name="condition", 
            value_name="beta")

        stim_angle = angles[voxel_dfs[t].condition.values.astype(int)]

        theta = np.asarray([convert_angle(a) for a in np.degrees(voxel_dfs[t].angle)])
        target_dist = fix_deg(theta - stim_angle)

        voxel_dfs[t].insert(9, 'pref_angle', theta)
        voxel_dfs[t].insert(10, 'stim_angle', stim_angle)
        voxel_dfs[t].insert(11, 'target_dist', target_dist)

        # Bin these distances
        dist_bins = np.arange(-180, 220, 20) - 10
        bins = pd.cut(voxel_dfs[t]['target_dist'], bins=dist_bins)
        bins = bins.apply(center_bin).astype(float)
        bins[bins == -180.0] = 180.0
        voxel_dfs[t].insert(12, 'ang_dist_bin', bins)

        voxel_dfs[t].insert(0, 'subj', '%03d' % wlsubj)
        
    angdist_df = pd.concat(voxel_dfs)

    print("Done!")
    
    return angdist_df


def get_angdist_DF(wlsubj, prevoxels, conds_by_trials, tasks, conds, angles, binsize = 20, query = ''):
    event_codes = prevoxels.columns.values[prevoxels.columns.str.isnumeric()]
    sacc_fname = '~/mnt/winawer/Projects/Interstellar/analyses/code/eyedata/sub-wlsubj%03d_saccades.tsv' % wlsubj
    sacc_fname = os.path.expanduser(sacc_fname)
    sacc = pd.read_csv(sacc_fname, sep = '\t', index_col = 0)
    sacc = sacc[sacc.sacc_label == 'response']
    df = []

    sacc['event_id'] = sacc.trial_id + ((sacc.run-1)*16)
    sacc['subj'] = wlsubj
    
    for e in event_codes:
        d = prevoxels.filter(['x', 'y', 'eccen', 'angle', 'sigma', 'vexpl', 'roi',
            'surf_label'])


        
        d.insert(0, 'subj', wlsubj)
        d.insert(1, 'task', tasks[conds_by_trials[int(e)]])
        d.insert(2, 'cond', conds_by_trials[int(e)])
        d.insert(3, 'event_id', int(e))
        d.insert(4, 'beta', prevoxels[e])

        df.append(d)
        
    df = pd.concat(df)

    V = df.merge(sacc, on = ['event_id', 'subj', 'task', 'cond'], how = 'outer')

    stim_angle = angles[V.cond.values.astype(int)]

    theta = np.asarray([convert_angle(a) for a in np.degrees(V.angle)])
    target_dist = fix_deg(theta - stim_angle)
    sacc_dist = fix_deg(theta - V.sac_ang)

    V.insert(4, 'pref_angle', theta)
    V.insert(5, 'stim_angle', stim_angle)
    V.insert(6, 'target_dist', target_dist)
    V.insert(7, 'sacc_dist', sacc_dist)

    # Bin these distances
    dist_bins = np.arange(-180, 220, binsize) - binsize/2
    bins = pd.cut(V['target_dist'], bins=dist_bins)
    bins = bins.apply(center_bin).astype(float)
    bins[bins == -180.0] = 180.0
    V.insert(8, 'ang_dist_bin', bins)

    dist_bins = np.arange(-180, 220, binsize) - binsize/2
    bins = pd.cut(V['sacc_dist'], bins=dist_bins)
    bins = bins.apply(center_bin).astype(float)
    bins[bins == -180.0] = 180.0
    V.insert(9, 'sacc_dist_bin', bins)

    return V



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
        gparams = {
            'beta': g['betasmd'],
            'R2': g['R2'],
            'se': g['glmbadness'],
            'rse': g['rrbadness']
        }

    return gparams

            
class IEM:
    '''
    S: Basis Set S (n_channels x n_discretepoints)
    data: list of voxelwise timeseries arrays (n_voxels x n_timepoints)
    design: list of design matrices (n_timepoints x n_conds)
    angles: array of angles used (n conditions x 1), 
        where the first angle = 1st column of design matrix, 
        2nd angle corresponds to 2nd column, etc. etc. 
    '''
    
    def __init__(self, S, data, design, angles):
        self.basis = S
        self.data = data
        self.design = design
        self.angles = angles
        self.W = np.zeros([self.data[0].shape[0], S.shape[0]])
        
        self.matrices = {"W": self.W}
    
    def info(self):
        print(
        ("IEM:\n-----------\n - N Channels: %d\n" +
            " - N Points per Channel: %d\n" +
            " - N Scans: %d\n" +
            " - N Voxels: %d\n" +
            " - TRs per scan: %d\n" +
            " - N Conditions: %d") % (
                self.basis.shape[0], 
                self.basis.shape[1], 
                len(self.design), 
                self.data[0].shape[0], 
                self.data[0].shape[1], 
                self.design[0].shape[1])
        )
        
        print("\nMatrix Shapes:")
        [print(" - %s: (%d, %d)" % (name, matrix.shape[0], matrix.shape[1])) for name, matrix in self.matrices.items()]
    
    def plot_channels(self):
        plt.plot(self.basis.T)
        
        
    def build_design(self, runs, width):
        T = np.concatenate([self.design[run] for run in runs], axis = 0)
        
        trials = T @ self.angles

        t = trials[trials != 0].astype(int)

        boxcar = np.ones(self.basis.shape[1])
        boxcar[width:-width] = 0

        A = scipy.linalg.toeplitz(boxcar)[:, t]

        C = self.basis@A
        
        return C
    
    
    def format_bold(self, start, stop, runs):
        B = []

        for i in runs:
            run = self.data[i]
            # Split this run into the 16 trials by voxels, averaged, for B training
            t_0 = np.where(self.design[i] == 1)[0]

            voxel_mean = np.expand_dims(run[:, :].mean(axis = 1), 1)
            voxel_diff = run - voxel_mean 
            S_delta = voxel_diff / voxel_mean * 100

            B.append(np.asarray([S_delta[:, t0+start:t0+stop] for t0 in t_0]).T.mean(axis = 0))

        B = np.concatenate(B, axis = 1)
        
        return B
    
    
    def train(self, runs, width, start, stop, verbose = 0):
        self.C = self.build_design(runs, width)
        self.B = self.format_bold(start, stop, runs)
        
        if verbose: print("Trained on Runs: %s" % (str(runs)))
        
        self.W = self.B @ self.C.T @ np.linalg.pinv(self.C @ self.C.T)
        
        self.matrices["W"] = self.W
        self.matrices["B"] = self.B
        self.matrices["C"] = self.C
        
        
    def test(self, runs, start, stop, verbose = 0):
        B_2 = []
        self.B_t = []

        for i in runs:
            t_0 = np.where(self.design[i] == 1)[0]

            run = self.data[i]
            
            voxel_mean = np.expand_dims(run[:, :].mean(axis = 1), 1)
            voxel_diff = run - voxel_mean
            S_delta = voxel_diff / voxel_mean * 100

            B_2.append(np.asarray([S_delta[:, t0+start:t0+stop] for t0 in t_0]).T.mean(axis = 0))
            b = self.design[i] @ self.angles
            b = b[b != 0].astype(int)
            for j in b: self.B_t.append(j)
            
        if verbose : print("Tested on Runs: %s" % (str(runs)))

        self.B_2 = np.concatenate(B_2, axis = 1)
        self.C_2 = np.linalg.pinv(self.W.T@self.W) @ self.W.T @ self.B_2
        
        self.matrices["B_2"] = self.B_2
        self.matrices["C_2"] = self.C_2
        
        
    def reconstruction(self, plot = False):
        self.R = self.C_2.T @ self.basis
        self.R_r = np.asarray(
            [np.roll(self.R[i], -self.B_t[i] + 180) for i in range(len(self.B_t))])
        
        self.R_avg = self.R_r.mean(axis = 0)
        
        if plot:
            plt.plot(self.R_avg)
            
        return self.R_avg
        

def bootstrap_data(data, wlsubjects, n=0):
    boot_subj = random.choices(wlsubjects, k = len(wlsubjects))

    boot_data = []
    for i, s in enumerate(boot_subj):
        df = data.query("subj==@s")
        df = df.assign(subj=i+1, orig_subj=s, n_boot = n)
        boot_data.append(df)
    
    boot_data = pd.concat(boot_data).reset_index(drop=True)
    
    return boot_data


def bootstrap_stream(stream):
    import pandas

    n, rng = stream
    boot_subj = rng.choice(subjects, len(subjects))

    boot_data = []
    for i, s in enumerate(boot_subj):
        df = roi_data.query("subj==@s")
        df = df.assign(subj=i+1, orig_subj=s, n_boot = n)
        boot_data.append(df)
    
    boot_data = pandas.concat(boot_data).reset_index(drop=True)
    
    return boot_data


def bootstrap_stream_split(stream):
    import pandas

    n, rng = stream
    boot_subj = rng.choice(subjects, len(subjects))

    boot_data_1 = []
    boot_data_2 = []
    for i, s in enumerate(boot_subj):
        df1 = roi_data_1.query("subj==@s")
        df1 = df1.assign(subj=i+1, orig_subj=s, n_boot = n)
        df2 = roi_data_2.query("subj==@s")
        df2 = df2.assign(subj=i+1, orig_subj=s, n_boot = n)
        boot_data_1.append(df1)
        boot_data_2.append(df2)
    
    boot_data_1 = pandas.concat(boot_data_1).reset_index(drop=True)
    boot_data_2 = pandas.concat(boot_data_2).reset_index(drop=True)
    
    return boot_data_1, boot_data_2


def norm_fit_vm(boot_data, n_TRs = 0):

    if n_TRs:
        p=[]
        for i in tqdm.tqdm(range(n_TRs)):
            if not i:
                n = norm_group(boot_data, yvar = str(i), group_cols = ['n_boot'])
            else:
                b = norm_group(boot_data, yvar = str(i), group_cols = ['n_boot']).filter(regex='adj')
                n.insert(TTA_n.shape[1], str(i) + '_adj', b)

            p.append(fit_diff_vonmises(n, str(i) + '_adj', group_cols = ['n_boot'], precision = False))

    else:
        n = norm_group(boot_data, group_cols = ['n_boot'])
        p = fit_diff_vonmises(n, 'beta_adj', group_cols = ['n_boot'], precision = False)     


    return n, p


def interstellar_bootstrap(data, n_boots, n_TRs = 0):
    N, F = [], []
    for n in tqdm.tqdm(range(n_boots)):
        df = bootstrap_data(data, n)
        norms, fits = norm_fit_vm(df, n_TRs)

        N.append(norms)
        F.append(fits)
    
    N = pd.concat(N)
    F = pd.concat(F)

    return N, F
    

def logistic(x, l, k, x0, c):
    y = l * (1 / (1 + np.exp(-k * (x - x0)))) + np.log(c)
    return y


def logistic2(x, l, k, x0, c, k2, x2, c2):
    y1 = l * (1 / (1 + np.exp(-k * (x - x0)))) + np.log(c)
    y2 = l * (1 / (1 + np.exp(-k2 * (x - x2)))) + np.log(c2)

    y = y1 * y2

    return y


def rect(x):
    x[x < 0] = 0

    return x

def ss(data, errs, last_pct = .2):
    y_sz = data.shape[0]
    y_idx = int(y_sz * (1 - last_pct))
    data = data[y_idx:].mean()

    e_sz = errs.shape[-1]
    e_idx = int(e_sz * (1 - last_pct))
    errs = errs[:, e_idx:].mean(axis = 1)

    return data, errs


def rect(x):
    x[x < 0] = 0

    return x

def get_GXM(x, y, pct_max, delta = 1):
    y1 = y.copy()[:y.argmax()]
    y1 -= y1.min()

    pct_y = y1.max() * pct_max

    point_idx = np.abs(y1 - pct_y).argmin()

    y_lower = y1[point_idx-delta]
    y_upper = y1[point_idx+delta]
    x_lower = x[point_idx-delta]
    x_upper = x[point_idx+delta]

    slope = (y_upper - y_lower) / (x_upper - x_lower)

    return slope, point_idx, pct_y


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


def boot_errbands(r, tsk, B_norm, TS_boots, t, conf_interval = [16, 84], ts = False):
    roi = r
    task = tsk
    bf = B_norm.query("roi == @roi & task == @task")
    TSb = TS_boots.query("roi_labels == @roi & task == @task")

    bfits = []
    b_timeseries = []
    for i, row, in bf.iterrows():
        b_ts = TSb.query("n_boot == @row.n_boot").amp.values
        b_ts = (b_ts-b_ts.min())/(b_ts.max() - b_ts.min())
        b_timeseries.append(b_ts)
        if task == 'wm':
            params = [*row.values[3:10]]
            y = logistic2(t, *params)
        else:
            params = [*row.values[3:7]]
            y = logistic(t, *params)

        bfits.append(y)
        
    bfits=np.vstack(bfits)
    err_bands = np.nanpercentile(bfits, conf_interval, axis=0)
    b_timeseries=np.vstack(b_timeseries)
    err_bands_ts = np.nanpercentile(b_timeseries, conf_interval, axis=0)

    return err_bands, err_bands_ts


def compute_boot_err(data, groupvar, yvar, boot_var = 'n_boot', conf_interval = [16, 84]):
    g_err = []
    for group, group_data in data.groupby(groupvar):
        y = group_data[yvar].values

        err = np.nanpercentile(y, conf_interval)

        g_err.append([group, *err])

    g_err = pd.DataFrame(g_err, columns = [groupvar, 'lower_ci', 'upper_ci'])
        
    return g_err

def interpolate_activity(data, basis = 'cartesian', normalize = False, offset = 0, yval = 'beta'):
    # Define stim space grid
    degs_lim = 12
    minval, maxval, stepval = [-degs_lim, degs_lim, .125]
    x, y = np.mgrid[minval:maxval:stepval, minval:maxval:stepval]
    
#     # Polar
#     thetaminval, thetamaxval, thetastepval = [-np.pi, np.pi, .0328]
#     rhominval, rhomaxval, rhostepval = [0, degs_lim, 0.125/2]
#     py, px = np.mgrid[thetaminval:thetamaxval:thetastepval, rhominval:rhomaxval:rhostepval]
    
#     # Log Polar
#     lthetaminval, lthetamaxval, lthetastepval = [-np.pi, np.pi, .0328]
#     lrhominval, lrhomaxval, lrhostepval = [0, np.log(degs_lim+0.0001), np.log(degs_lim+0.0001)/192]
#     ly, lx = np.mgrid[lthetaminval:lthetamaxval:lthetastepval, lrhominval:lrhomaxval:lrhostepval]
    
    
    ang_dists_rads = np.radians(data.target_dist + offset)
    rot_x, rot_y = pol2cart(data.eccen, ang_dists_rads)
    
    data['rot_x'] = rot_x.values
    data['rot_y'] = rot_y.values
    data['theta'] = ang_dists_rads
    data['rho'] = data.eccen
    data['lrho'] = np.log(data.eccen+0.0001)
    
    if basis == 'cartesian':
        pts = data[['x', 'y']].values
        vals = data[yval].values
        sinterp = griddata(pts, vals, (x, y), method = 'linear', rescale = False).T

        pts = data[['rot_x', 'rot_y']].values
        vals = data[yval].values
        sinterp_rot = griddata(pts, vals, (x, y), method = 'linear', rescale = False).T

    if normalize:
        sinterp = (sinterp - np.nanmean(sinterp)) / np.nanstd(sinterp)
        sinterp_rot = (sinterp_rot - np.nanmean(sinterp_rot)) / np.nanstd(sinterp_rot)
        
    sdf = pd.DataFrame(dict(
        # stim_angle = stim_angle, 
        activity_map=sinterp.flatten(), 
        activity_map_rot=sinterp_rot.flatten(), 
        task = data.task.unique()[0], 
        inds=np.arange(len(sinterp.flatten()))))
    
    return sdf.reset_index(drop = True), data
      

def avg_activity_figure(interp_data, vmax=1.5, degs_lim = 8, tasks = ['perception', 'ltm', 'wm'], 
                        colors = ['b', 'orange', 'g'], stim_radius = 0, cmap = 'RdBu_r', plot = True):
    
    # Average across subjects 
    dplot = interp_data.groupby(['task', 'inds'])['activity_map_rot'].aggregate(np.nanmedian).reset_index()
    # Create axes and get data for each facet

    with sns.axes_style("white"):
        g = sns.FacetGrid(col='task', 
                         height=3, aspect=1, sharex=False, data=interp_data)
    ax_data = []
    i = 0
    
    for _, d in g.facet_data():
        s = tasks[i]
        i += 1
        ax_data.append(dplot[dplot.task == s]['activity_map_rot'].values)
        
        
        
    # Plot heatmap on each facet
    ax_heatmap = activity_heatmap(ax_data, g, degs_lim = degs_lim, 
                     vmax=vmax, stim_radius = stim_radius,
                     colors = colors, titles = tasks, cmap = cmap, plot = plot)
    
    return ax_heatmap
    

def activity_heatmap(ax_data, g, degs_lim = 8, vmax=1.5, stim_radius = 0, colors = ['b', 'o', 'g'], titles = [], cmap = 'RdBu_r', plot = True, plot_stim_circle = False):
    n_steps = len(np.arange(-degs_lim, degs_lim, 0.125))
    
    for i, d in enumerate(ax_data):
        ax = g.axes.flatten()[i]
        
        ax_heatmap = np.reshape(d, (n_steps, n_steps))
        
        if plot:
            sns.heatmap(ax_heatmap, cbar=False, square=True, cmap=cmap, 
                        vmin=-1*vmax, vmax=vmax, linewidths=0, cbar_kws={'ticks':[]}, ax=ax)
            ax.invert_yaxis()
            #ax.set(xlim=[-4, 132], ylim=[-4, 132])
            ax.set_xticklabels(labels=[], rotation=0)
            ax.set_yticklabels(labels=[])
            ax.axhline(y=n_steps/2, c="0")
            ax.axvline(x=n_steps/2, c="0")
            ax.text(n_steps+8, n_steps/2, "$\it{x}$", ha='left', va='center', color="0", size=12)
            ax.text(n_steps/2, -20, "$\it{y}$", ha='center', va='bottom', color="0", size=12)

            ax.set_title(titles[i].title())
            if plot_stim_circle:
                circle2 = plt.Circle((n_steps/2, n_steps/2), n_steps/2 * (stim_radius/degs_lim), color=colors[i], fill=False, lw = 2)
                ax.add_patch(circle2)
    
    if plot:
        sns.despine(trim=True, left=True, bottom=True)
        plt.gcf().tight_layout(h_pad=3)

    return ax_heatmap
    
    
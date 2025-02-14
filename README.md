# Woodry_Curtis_Winawer_2025

Paper: [Feedback scales the spatial tuning of cortical responses during visual working memory and long-term memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC11042180/)

Github: [https://github.com/rwoodry/Woodry_Curtis_Winawer_2025](https://github.com/rwoodry/Woodry_Curtis_Winawer_2025)

OSF: [https://osf.io/4mf8j/](https://osf.io/4mf8j/)

OpenNeuro:

## Paper Figures

The scripts used to generate the figures in the paper are located in /paper/figures. There is a jupyter notebook and a python script for each one. These scripts rely on dataframes (.tsv files) stored in /derivatives/dataframes.

Dataframes necessary for each figure:

| **Figure** | Dataframes |
| --- | --- |
| 4 | /glmsingle/allsubj_vmfits_ang_dist_bin.tsv /glmsingle/bootstrap_vmfits/allsubj_vmfits_ang_dist_bin_batch_{1:20}.tsv /glmsingle/allsubj_dist_normedbetas_ang_dist_bin.tsv |
| 5 | /glmsingle/allsubj_vmfits_sacc_ang_dist_bin_saccade-aligned.tsv /glmsingle/allsubj_vmfits_ang_dist_bin_target-aligned_comparison.tsv /glmsingle/allsubj_vmfits_ang_dist_bin_{7, 9, 12}_1.tsv |
| 6 | /p-ltm-wm-cue-sacc-wmtarget/sub-wlsubj{XXX}/timeseries/denoiseddata_TTA_run-{01:100}.tsv /p-ltm-wm-cue-sacc-wmtarget/bootstrap_logfits/tta_logfits_batch_{1:20}.tsv |
| 7 | /saccades/sub-all_saccades.tsv /saccades/trialwise_saccades.tsv /glmsingle/allsubj_vmfits_ang_dist_bin_saccsplit.tsv |
| 8 | /glmsingle/allsubj_vmfits_ang_dist_bin.tsv /glmsingle/bootstrap_vmfits/allsubj_vmfits_ang_dist_bin_batch_{1:20}.tsv /favila_natcomms_2022_tuning.tsv /glmsingle/allsubj_vmfits_ang_dist_bin_nearmidfar.tsv /glmsingle/bootstrap_nearmidfar/allsubj_vmfits_ang_dist_bin_nearmidfar_batch_{1:20}.tsv |

***Brackets {} are placeholders which denote multiple values, and therefore refer to multiple dataframes.

## Code & Processed Data

To download

/figures/requirements_figure.txt lists the python dependencies required for the figure scripts to run properly

requirements_pip.txt and requirements_pip.txt each lists all the dependencies featured in the conda environment used for the entire project. One using pip and the other using conda. Many of these are likely not necessary. Still, the project environment dependencies are listed if needed.

Processed data lives in /derivatives/dataframes. These are obtained using several scripts found in /code. These are ran in order according to code_order.txt. 

/code/code_order.txt outlines what each step does, and the terminal calls used to execute those steps. 

**Note:** Several of the steps listed in code_order use sbatch calls. This is because these steps were processed at NYU’s Greene High Performance Compute cluster. Each of these sbatch calls rely on a particular script in /code.

**Note:** The steps that create the timeseries dataframes require outputs from [GLMdenoise](https://github.com/cvnlab/GLMdenoise). These outputs are included in the data repository. However, if you want to reproduce the GLMdenoise ouptuts from the  BOLD timeseries data, you will need the GLMdenoise library (found [here](https://github.com/cvnlab/GLMdenoise))

## Dataframes

For each dataframe used in for generating the figures, the column variables and their descriptions are listed below. The variable names and meanings will be similar for other dataframes not listed. Please reach out to Rob Woodry (rfw256 at nyu dot edu) for more information.

### Figure 4

**/glmsingle/allsubj_vmfits_ang_dist_bin.tsv**

| **Column Name** | **Info** |
| --- | --- |
| model | GLM used |
| roi_labels | Region of interest label (V1, V2, LO1, etc.) |
| task | Experiment task (perception, working memory: ‘wm’, long-term memory: ‘ltm’) |
| func | Function used to fit polar angle activation profile (only difference of Von Mises) |
| loc | Center of Von Mises in distance from target (in radians) |
| loc_deg | Center of Von Mises in distance from target (in degrees) |
| kappa1 | Kappa parameter of first Von Mises |
| scale1 | Scale parameter of first Von Mises |
| kappa2 | Kappa parameter of second Von Mises |
| scale2 | Scale parameter of second Von Mises |
| maxr | Maximum response of Difference of Von Mises |
| minr | Minimum response of Difference of Von Mises |
| amp | Amplitude of Difference of Von Mises (Peak to trough; maxr - minr) |
| fwhm | Full width at half maximum amplitude of Difference Von Mises |

**allsubj_vmfits_ang_dist_bin_batch_{1:20}.tsv**

Same as */glmsingle/allsubj_vmfits_ang_dist_bin.tsv* (above), but with one additional column:

| **Column Name** | **Info** |
| --- | --- |
| n_boot | Bootstrap id. There are 500 boots per group, so this value ranges from 0 to 499. |

**glmsingle/allsubj_dist_normedbetas_ang_dist_bin.tsv**

| **Column Name** | **Info** |
| --- | --- |
| model | GLM used |
| roi_labels | Region of interest label (V1, V2, LO1, etc.) |
| task | Experiment task (perception, working memory: ‘wm’, long-term memory: ‘ltm’) |
| ang_dist_bin | Polar angle distance bin group. Varies from -160 to 180 (in degrees) |
| wlsubj | Subject id. This column is meaningless after averaging across subjects |
| beta | The beta values obtained from the GLM. Here they are binned by ang_dist_bin and averaged |
| vexpl | Variance explained obtained from the pRF fits. Here they are binned by ang_dist_bin and averaged |
| norm | The vector length of the beta values across ang_dist_bins.  |
| beta_norm | Values in ‘beta’ normalized by ’norm’ |
| norm_perc | The vector length if the beta values across ang_dist_bin from the perception task |
| beta_norm_perc | Values in ‘beta’ normalized by ‘norm_perc’ |
| beta_adj | The average ‘beta_norm’ values, rescaled by average ‘norm’ |
| beta_adj_perc | The average ‘beta_norm_perc’ values, rescaled by average ‘norm_perc’ |

### Figure 5

**/glmsingle/allsubj_vmfits_sacc_ang_dist_bin_saccade-aligned.tsv**

Same as Figure 4’s */glmsingle/allsubj_vmfits_ang_dist_bin.tsv* above, but where the fits are obtained from polar angle activation profiles aligned to the saccade location, not target location.

**/glmsingle/allsubj_vmfits_ang_dist_bin_target-aligned_comparison.tsv**

Same as Figure 4’s */glmsingle/allsubj_vmfits_ang_dist_bin.tsv* above

**/glmsingle/allsubj_vmfits_ang_dist_bin_{7, 9, 12}_1.tsv**

Same as Figure 4’s */glmsingle/allsubj_vmfits_ang_dist_bin.tsv* above, but where voxels are selected assuming different target eccentricities (7, 9, 12).

### Figure 6

**/p-ltm-wm-cue-sacc-wmtarget/sub-wlsubj{XXX}/timeseries/denoiseddata_TTA_run-{01:100}.tsv**

| **Column Name** | **Info** |
| --- | --- |
| roi_labels | Region of interest label (V1, V2, LO1, etc.) |
| task | Experiment task (perception, working memory: ‘wm’, long-term memory: ‘ltm’) |
| ang_dist_bin | Polar angle distance bin group. Varies from -160 to 180 (in degrees) |
| trialnum | Trial number. Column leftover after trial-triggered average, and is therefore meaningless here |
| pref_angle | average preferred polar angle (in degrees) of voxels used for this ang_dist_bin   |
| stim_angle | Stimulus angle. Column leftover after trial-triggered average, and is therefore meaningless here |
| target_dist | Average angular distance between voxel’s preferred angle and stim_angle. Averages out to very close to ang_dist_bin IDs |
| roi_int_labels | ROI integer labels used for drawing and saving ROI outlines on the surface |
| x | average preferred x location (in degrees) of voxels used for this ang_dist_bin   |
| y | average preferred y location (in degrees) of voxels used for this ang_dist_bin   |
| eccen | average preferred eccen (in degrees) of voxels used for this ang_dist_bin   |
| angle | average preferred angle (in radians) of voxels used for this ang_dist_bin   |
| sigma | average pRF size of voxels used for this ang_dist_bin   |
| vexpl | average pRF variance xplained of voxels used for this ang_dist_bin   |
| -3: 16 | These columns refer to the average activity at T time steps relative to cue onset in the trial.  |

**/p-ltm-wm-cue-sacc-wmtarget/bootstrap_logfits/tta_logfits_batch_{1:20}.tsv**

| **Column Name** | **Info** |
| --- | --- |
| n_boot | Bootstrap ID. Ranges from 0-499 |
| roi | Region of interest label (V1, V2, LO1, etc.) |
| task | Experiment task (perception, working memory: ‘wm’, long-term memory: ‘ltm’) |
| l | The upper asymptote of the amplitude estimates |
| k | the growth rate of the logistic function |
| x0 | Midpoint of logistic function |
| c | baseline of the logistic function |
| k2 | the growth rate of the logistic function (only for working memory) |
| x2 | Midpoint of the second logistic function (only for working memory) |
| c2 | baseline of the second logistic function (only for working memory) |
| risetime | Time to rise to %90 |
| risetime_l | Time to rise to %90 for Working Memory logistic |
| fwhm_t00:19 | These columns refer to the FWHM of the binned activations at T time steps relative to cue onset in the trial.  |

### Figure 7

**/saccades/sub-all_saccades.tsv**

| **Column Name** | **Info** |
| --- | --- |
| wlsubj | subject ID |
| run | run number |
| task | Experiment task (perception, working memory: ‘wm’, long-term memory: ‘ltm’) |
| trial_id | Trial number |
| cond | condition number (maps to specific target position for task) |
| sacc_idx | Saccade index within trial |
| sacc_label | Saccade classification (first, corrective, response). Response is saccade closest to target eccentricity. |
| index | Eyelink event index |
| event | Eyelink event code |
| eye | Recorded eye (R: right) |
| start_time | start time of saccade relative to start of experiment in milliseconds |
| end_time | end time of saccade relative to start of experiment in milliseconds |
| duration | duration of saccade in milliseconds |
| x_start | X coordinate of saccade start position in screen coordinates |
| y_start | Y coordinate of saccade start position in screen coordinates |
| x_end | X coordinate of saccade end position in screen coordinates |
| y_end | Y coordinate of saccade start position in screen coordinates |
| amp_deg | Saccade amplitude in degrees |
| peak_v | Saccade maximum velocity |
| trial_start | Start time of trial relative to start of experiment (in milliseconds) |
| trial_end | End time of trial relative to start of experiment (in milliseconds) |
| target_x | Trial target x location (in degrees) |
| target_y | Trial target y location (in degrees) |
| sac_x_deg | Saccade x location in degrees |
| sac_y_deg | Saccade y location in degrees |
| sac_ang | Saccade angular location in degrees |
| sac_ecc | Saccade eccentricity in degrees |
| target_theta | Target angular location in radians |
| sacc_theta | Saccade angular location in radians |
| ang_dist | Angular distance b/w saccade and target location |
| fix_duration | Duration of fixation in milliseconds |
| x_centered | Centered x position  |
| y_centered | Centered y position |

**/saccades/trialwise_saccades.tsv**

| **Column Name** | **Info** |
| --- | --- |
| subj | subject ID |
| task | Experiment task (perception, working memory: ‘wm’, long-term memory: ‘ltm’) |
| event_id | Trial number with respect to all runs |
| tertile | Trial’s saccade tertile group assignment (clock:clockwise, counter: counterclockwise, center: near target location) |
| ang_dist | Saccade angular distance to target location (in degrees) |

**/glmsingle/allsubj_vmfits_ang_dist_bin_saccsplit.tsv**

Same as */glmsingle/allsubj_vmfits_ang_dist_bin.tsv* from Figure 4, but with another column:

| **Column Name** | **Info** |
| --- | --- |
| tertile | Saccade tertile group assignment (clock:clockwise, counter: counterclockwise, center: near target location) |

### Figure 8

*/glmsingle/allsubj_vmfits_ang_dist_bin.tsv, /glmsingle/bootstrap_vmfits/allsubj_vmfits_ang_dist_bin_batch_{1:20}.tsv*

are same as in Figure 4, above.

**/glmsingle/allsubj_vmfits_ang_dist_bin_nearmidfar.tsv
/glmsingle/bootstrap_nearmidfar/allsubj_vmfits_ang_dist_bin_nearmidfar_batch_{1:20}.tsv**

Are the same as */glmsingle/allsubj_vmfits_ang_dist_bin.tsv, /glmsingle/bootstrap_vmfits/allsubj_vmfits_ang_dist_bin_batch_{1:20}.tsv* in Figure 4, respectively, with an extra column:

| **Column Name** | **Info** |
| --- | --- |
| nearest_dist_bin | Stimulus group assignment based on distance to nearest neighboring stimulus (near, mid, far) |

**/favila_natcomms_2022_tuning.tsv**

| **Column Name** | **Info** |
| --- | --- |
| metric | Spatial tuning metric (loc_deg: Peak location, amp: amplitude, fwhm: tuning width) |
| roi_labels | Region of interest label (V1, V2, LO1, etc.) |
| task | Experiment task (perception, working memory: ‘wm’, long-term memory: ‘ltm’) |
| value | Metric value |
| lower_68_CI | Lower 68% Confidence Interval |
| upper_68_CI | Upper 68% Confidence Interval |
| lower_95_CI | Lower 95% Confidence Interval |
| Upper_95_CI | Upper 95% Confidence Interval |

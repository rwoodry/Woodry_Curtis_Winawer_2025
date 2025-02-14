# %%
import numpy as np
import os
from glmsingle.glmsingle import GLM_single
from glob import glob
import pandas as pd
import nibabel as nib
from pprint import pprint
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--subject', required = True, type = int)
    parser.add_argument('--derivatives_dir', required = True)

    args = parser.parse_args()

    subj = args.subject
    deriv_dir = args.derivatives_dir
    stimdur = 11.5
    tr = 1

    print(subj, deriv_dir)

    data_dir = os.path.join(deriv_dir, "fsnative_both/sub-wlsubj%03d" % subj)
    design_dir = os.path.join(deriv_dir, "design_matrices/glmsingle/sub-wlsubj%03d" % subj)

    session_indicator = [1,1,1,1,1,1,2,2,2,2,2,2]

    data = []
    designs = []

    for run in range(12):
        data_fname = os.path.join(data_dir, "run_%02d.tsv" % run)
        design_fname = os.path.join(design_dir, "sub-wlsubj%03d_run-%02d.tsv" % (subj, run))

        run_data = pd.read_csv(data_fname, sep = '\t', index_col = 0, header = None)
        design = pd.read_csv(design_fname, sep = '\t', index_col = 0, header = None)

        print(run_data.shape, design.shape)

        data.append(np.asarray(run_data))
        designs.append(np.asarray(design))

    opt = dict()
    # set important fields for completeness 
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [1,1,1,1]
    opt['wantmemoryoutputs'] = [1,1,1,1]

    opt['sessionindicator'] = session_indicator

    # Create a GLM_single object
    glmsingle_obj = GLM_single(opt)

    # visualize all the hyperparameters
    pprint(glmsingle_obj.params)

    # %%
    # create a directory for saving GLMsingle outputs
    outputdir_glmsingle = os.path.join(deriv_dir, 'GLMsingle', 'sub-wlsubj%03d' % subj)
    figure_dir = os.path.join(outputdir_glmsingle, 'figures')
    if not os.path.exists(outputdir_glmsingle):
        print("Creating output directory located at:", outputdir_glmsingle)
        os.makedirs(outputdir_glmsingle, 0o666)

    if not os.path.exists(figure_dir): 
        os.makedirs(figure_dir)


    
    print(len(designs), len(data))

    print('running GLMsingle...')

    # run GLMsingle
    results_glmsingle = glmsingle_obj.fit(
        designs,
        data,
        stimdur,
        tr,
        outputdir=outputdir_glmsingle,
        figuredir=figure_dir)

if __name__ == "__main__":
    main()





addpath(genpath('GLMdenoise'));
% addpath(genpath('glmdenoise_data'));
% addpath(genpath('glmdenoise_designs'));

subj = 135;
stimdur = 1;
tr = 1;
n_runs = 12;

designs = cell(n_runs, 1);
data = cell(n_runs,1);

for run = 1:n_runs
    design_fname = sprintf("glmdenoise_designs/sub-wlsubj%03d/run_%02d.tsv", subj, run-1);
    design = readtable(design_fname, "FileType","text",'Delimiter', '\t');
    designs{run} = design{:, 2:end};
    disp(size(designs(run)));

    data_fname = sprintf("glmdenoise_data/sub-wlsubj%03d/run_%02d.tsv", subj, run-1);
    data_matrix = readtable(data_fname, "FileType","text",'Delimiter', '\t');
    data{run} = data_matrix{:, 2:end};
    disp(size(data(run)));
end

[results, denoiseddata] = GLMdenoisedata(design,data,stimdur,tr,[],[],[],'glm_figures');

save(sprintf('results_sub-wlsubj%03d.mat', subj),'results');
save(sprintf('denoiseddata_sub-wlsubj%03d.mat', subj),'denoiseddata');
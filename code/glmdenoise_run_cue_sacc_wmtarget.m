addpath(genpath('GLMdenoise'));
addpath(genpath('glmdenoise_data'));
addpath(genpath('glmdenoise_designs'));
addpath(genpath('glmdenoise_output'));


stimdur = 1;
tr = 1;
n_runs=12;

% model = 'p-ltm-wm'
% model = 'p-ltm-wm-cue-sacc'
model = 'p-ltm-wm-cue-sacc-wmtarget'

n_runs=12;

disp(sprintf("Subject id: wlsubj%03d", subj));
disp(sprintf("n runs: %02d", n_runs));

designs = cell(1, n_runs);
data = cell(1, n_runs);

design_fnames = dir(sprintf("/scratch/rfw256/Interstellar/derivatives/design_matrices/%s/sub-wlsubj%03d/sub*", model, subj));

% Make dirs if none exist
subj_dir = sprintf('/scratch/rfw256/Interstellar/derivatives/GLMdenoise/%s/sub-wlsubj%03d', model, subj)

if ~exist(sprintf("/scratch/rfw256/Interstellar/derivatives/GLMdenoise/%s", model))
    mkdir(sprintf("/scratch/rfw256/Interstellar/derivatives/GLMdenoise/%s", model))
end

fig_dir = sprintf('/scratch/rfw256/Interstellar/derivatives/GLMdenoise/%s/sub-wlsubj%03d/glm_figures', model, subj)
if ~exist(subj_dir, 'dir')
    
    mkdir(subj_dir)
    mkdir(fig_dir)
end

for run = 1:n_runs
    design_fname = sprintf("/scratch/rfw256/Interstellar/derivatives/design_matrices/%s/sub-wlsubj%03d/sub-wlsubj%03d_run-%02d.tsv", model, subj, subj, run-1);
    design = readtable(design_fname, "FileType","text",'Delimiter', '\t');
    designs{run} = design{:, 2:end};
    disp(design_fname);
    disp(size(designs{run}));

    data_fname = sprintf("/scratch/rfw256/Interstellar/derivatives/fsnative_both/sub-wlsubj%03d/run_%02d.tsv", subj, run-1);
    data_matrix = readtable(data_fname, "FileType","text",'Delimiter', '\t');
    data{run} = data_matrix{:, 2:end};
    disp(data_fname);
    disp(size(data{run}));
end


[results, denoiseddata] = GLMdenoisedata(designs,data,stimdur,tr,[],[],[],fig_dir);

fn = fieldnames(results);

disp('Saving glm outputs...')
for i = 1:numel(fn)
    try
        file_to_save = results.(fn{i});

        if iscell(results.(fn{i}))
            for j = 1:numel(results.(fn{i}))
                disp(sprintf("%s %d", fn{i}, j));
                file_to_save = results.(fn{i}){j};
                a = size(size(file_to_save));
                
                if a(2) > 2
                    save(sprintf(strcat(subj_dir, '/%s_%d.mat'), fn{i}, j), 'file_to_save', "-v7.3");
            
                else

                    fname = sprintf(strcat(subj_dir, '/%s_%d.txt'), fn{i}, j);
                    writematrix(file_to_save, fname, 'Delimiter', 'tab');
                end
            end
        else

            disp(fn{i});
            fname = sprintf(strcat(subj_dir, '/%s.txt'), fn{i});
            writematrix(file_to_save, fname, 'Delimiter', 'tab');
        end
    catch
        disp("ERROR");
        % Nothing to do
    end

end

disp("Saving denoised data...");
for i = 1:numel(denoiseddata)
    file_to_save = denoiseddata{i};
    fname = sprintf(strcat(subj_dir, '/denoised-timeseries_run-%02d.txt'), i);
    disp(fname);
    writematrix(file_to_save, fname, 'Delimiter', 'tab');
end
%% Measuring Strength of Joint Causal Effects
% This code reproduces the figures/tables from our paper. To generate all figures 
% (as .png files), you just need to run main.m. The code should run with no
% issues using Matlab 2022a or later. All generated figures and tables will
% be saved to the results folder.

%% Set up the Matlab path
addpath(genpath('./data'))
addpath('./functions')
addpath('./results')
addpath('./scripts')


%% Record to a log file
diary './results/output.log'

% Optional: Turn off warnings (that might clutter the log file)
% warning('off','all')

%% Experiments
% Running these scripts will reproduce figures from the paper

% Table 1
disp('Generating Table 1')
table_kernels

% Fig 1
% Visualization of the local interaction function
disp('Generating Figure 1')
fig_local_int_fcn

% Fig 2
% Comparison of kernels
disp('Generating Figure 2')
Fig_kernels

% Figs 3 and 4
%   Figure 3: A sample from the Volterra model
%   Figure 4: Comparison of the true model, the estimated MDCE from GPR,
%   the Volterra model estimate, and the Bayes detector
disp('Generating Figures 3 and 4')
Fig3_4_Volterra

% Fig 5
% Robustness of the MDCE to linear confounders
disp('Generating Figure 5')
Fig5_confounder

% Fig 6
% Comparison of exact GPR with sparse approximation
disp('Generating Figure 6')
Fig6_sparse

% Fig 7
% Real data example using New Taipei City data
disp('Generating Figure 7')
fig_taipei


%% Stop writing to the log file
disp('Running is complete')
diary 'off'


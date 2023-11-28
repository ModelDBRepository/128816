%% example RF model, with explanation
clear all; close all

% Inter-cluster
Nc = 20;            % number of clusters
Pc = 0.1;           % probability of cluster-unit connection for long-range input 
r = 1;              % power-law exponent for inter-cluster connection probability; or probability of connection in spatially-uniform model (0 < r < 1)

% Intra-cluster
nn = 50;            % number of units per cluster
P = 0.7;            % proportion of projection units per cluster
C = 0.1;            % probability of connection for each local interneuron
Sp = 1;             % all projection neurons receive sensory input
Si = 0;           % proportion of interneurons receiving sensory input
Ip = 0;           % proportion of inhibitory projection neurons
Ii = 1;             % proportion of inhibitory interneurons

% simulation parameters
max_steps = 1;      % to just get structure;
max_steps = 1e4;   % to run dynamic simulation; maximum number of time-steps
con = 1e-4;         % convergence criteria
theta = 0;          % unit threshold
seed = 1;           % random number generator seed
flag = 'fi'; W = 0.1;  % probably point attractor model; constant weight for all connections: note that this sets it for +ve connections as flag 'i' is also selected
flag = 'fi'; W = 0.5;  % probably oscillatory model; constant weight for all connections: note that this sets it for +ve connections as flag 'i' is also selected


% input
S = zeros(Nc,1);    % per-cluster input
S(1) = 0.5;
S(8) = 0.5;

% pattern = zeros(Nc * nn,1);
% pattern(50) = 1;

% rand('state',2);
% per = 0.01;
%pattern = zeros(Nc*nn,1);
% pattern(rand(Nc*nn,1) <= per) = 1;
% S = pattern;

%% build model, and run as network if required
% NOTES: to just assess RF model's structure, set NC, C, Pc, P, and r to
% required values and MAX = 1 so that simulation is not run. 
[clust_out,clust_act,Proj_units,steps_elapsed,Clust_con,S_clust,proj_out,samp] = discrete_cluster1(Nc,nn,C,Pc,Sp,Si,P,Ip,Ii,r,con,max_steps,theta,S,seed,flag,[],[],W);

%% show output
if steps_elapsed > 1
    [total_clust_out clust_order] = sort(sum(proj_out));
    fliplr(clust_order) % so that highest output is first on list

    all_out = reshape(clust_out,nn,Nc);

    %%% plot of all activity
    figure
    pcolor(samp)
    shading('flat')

    %%% time-series from a random sample of 5% of neurons
    T = Nc*nn;
    idxs = ceil(rand(round(T*0.05),1) * T);
    idxs = sort(idxs); tseries = samp(idxs,1:end)';
    figure
    plot(tseries); xlabel('Time'); ylabel('Activity'); 
    axis([0 steps_elapsed -0.05 1.05])
end


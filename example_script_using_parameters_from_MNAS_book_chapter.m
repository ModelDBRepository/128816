%   example script to set up and run RF cluster model as used in the MNAS book chapter; some examples of
%   visualisation of results are included. 
%
%   Mark Humphries 19/6/2008

clear all
cl_fig
%%%%%%%%%%%% define general structural model parameters %%%%%%%%%%%
% Inter-cluster
Nc = 8;            % number of clusters
Pp = 0.1;           % probability of cluster-unit connection for long-range input 
Pc = 0.25;              % power-law exponent for inter-cluster connection probability (or P(c) for spatially uniform model) 

% Intra-cluster
n = 100;             % number of units per cluster
rho = 0.8;          % proportion of projection units per cluster
Pl = 0.25;           % 0.1..probability of connection for each local interneuron
rho_s = 1;             % all projection neurons receive sensory input
lambda_s = 1;           % proportion of interneurons receiving sensory input
rho_i = 0;             % proportion of inhibitory projection neurons
lambda_i = 1;             % proportion of inhibitory interneurons

%%%%%%%%%%% parameters for full model   %%%%%%%%%%%%%%%%%%%%%%%%%
% simulation parameters
con = 1e-6;         % convergence criteria
max_steps = 10e3;            % maximum number of time-steps
%max_steps = 1;              % set to 1 to get structure only   
theta = -0.05;          % unit threshold
slope = 1;                  % output slope
seed = 1;                   % random number generator seed

flag = 'if';         % flag for options - see DISCRETE_CLUSTER1 help

% set weights...
W = 0.1;    % low weight (0.1) for stable attractor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%% input parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = zeros(Nc,1);  % per-cluster input
rand('state',3)
% input to single clusters
S(3) = 0.4;
% S(4) = 0.3; % competing cluster inputs
% 
% % random input to each cluster
% S = rand(Nc,1);

% random inputs to all projection units...
%flag = [flag 'p'];
%S = exprnd(0.05,Nc*n,1); % most get small, some get large

%%%%%%%%%%%%%%% RUN FULL MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[clust_out,clust_act,Proj_units,steps_elapsed,Clust_con,S_clust,proj_out,samp,t_samp,this_model] =...
    discrete_cluster1(Nc,n,Pl,Pp,rho_s,lambda_s,rho,rho_i,lambda_i,Pc,con,max_steps,theta,S,seed,flag,[],[],W);

[total_clust_out clust_order] = sort(sum(proj_out));
[Ss,in_order] = sort(S');
Corder = fliplr(clust_order) % so that highest output is first on list
Sorder = fliplr(in_order);

% output on final time-step 
all_out = reshape(clust_out,n,Nc);

% get projection unit indices
[r p_per_cluster] = size(Proj_units);
temp_p_units = zeros(Nc,p_per_cluster);
for loop1 = 1:Nc
    temp_p_units(loop1,:) = Proj_units(loop1,:) + (loop1-1) * n;
    strLgd{loop1} = num2str(loop1);
end
temp_p_units = temp_p_units';
idx_p_units = sort(temp_p_units(:));

% get interneuron indices
idxCells = 1:n*Nc;
idx_i_units = setdiff(idxCells,idx_p_units)';

%% calculate mean projection neuron output per cluster%%
[r t] = size(samp);
p_out = samp(idx_p_units,:);

temp = reshape(p_out,p_per_cluster,t*Nc);
mean_clust_out = reshape(mean(temp),Nc,t); % mean cluster output is now in each row
figure(1)
plot(t_samp,mean_clust_out')
title('Mean cluster output from Full model')
xlabel('Time-step')
ylabel('Mean output')
legend(strLgd,'Location','Best')

%% look at distributions of final output...
figure(2); clf
hist(proj_out(:),30); xlabel('Output at end of simulation'); ylabel('Number of neurons')

%% look at dists per cluster
% [hist_clust,x] = hist(proj_out,20);
% figure(3)
% plot(x,hist_clust); xlabel('Total cluster output at end of simulation'); ylabel('Number of neurons')


%%% views of all neurons
figure(4)
pcolor(p_out)
shading('flat')
colorbar
title('Outputs of projection neurons')
xlabel('Time-step')
ylabel('Cell #')
  
i_out = samp(idx_i_units,:);
figure(5)
pcolor(i_out)
shading('flat')
colorbar
title('Outputs of inter-neurons')
xlabel('Time-step')
ylabel('Cell #')

%%% structure of network - only use this for small networks (n*Nc < 200)!!
% visualise_net(Clust_con,'directed')

%%%%%%%%%% analyse oscs and corrs
% if steps_elapsed >= max_steps
%     % then didn't terminate through equilibrium
%     %%%%%% analyse oscillations and correlations %%%%%%%%%%%%%%%%%%
%     Bpout = p_out(:,floor(steps_elapsed/2):end);   % remove transient start-up section
% 
%     % get fluctuation stats....
%     std_p = std(Bpout');
%     mean_p = mean(Bpout');
%     CV_p = std_p./mean_p;
% 
%     %% after transient, so:
%     % all CV = NaN = no output
%     % all CV = 0 = constant output
%     ix_zero = find(isnan(CV_p));
%     ix_const = find(CV_p==0);
%     ix_osc = find(~isnan(CV_p) & (CV_p>0));
% 
%     %% do pair-wise correlation between all osc outputs...
%     cor = corrcoef(Bpout(ix_osc,:)');
% 
%     %% find their parent cluster
%     osc_p_idxs = idx_p_units(ix_osc);
%     cluster_id = ceil(osc_p_idxs / n);  % get cluster membership
% 
%     %% plot "strong" correlations
%     tempP = cor > 0.8;
%     tempAP = cor < -0.8;
% 
%     visualise_groups(tempP,'undirected',cluster_id)
%     visualise_groups(tempAP,'undirected',cluster_id)
% end
% 




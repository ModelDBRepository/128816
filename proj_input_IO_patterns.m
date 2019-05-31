%   for MNAS book chapter (Humphries et al, 2010); Fig 5a,b
%   script to test relationship between input-output patterns in mRF model: 
%   Inputs: vectors of projection neuron input
%   Outputs: all projection neuron outputs; total cluster output
%   Matches: correlation between input-output orthogonality
%
%   Mark Humphries 30/4/2009

clear all
cl_fig

n_ins = 50;    % number of input vectors....

%%%%%%%%%%%% define general structural model parameters %%%%%%%%%%%
% Inter-cluster
Nc = 8;            % number of clusters
Pp = 0.1;           % probability of cluster-unit connection for long-range input 
Pc = 0.25;              % power-law exponent for inter-cluster connection probability (or P(c) for spatially uniform model) 

% Intra-cluster
n = 50;             % number of units per cluster
rho = 0.8;          % proportion of projection units per cluster
Pl = 0.25;           % 0.1..probability of connection for each local interneuron
rho_s = 1;             % all projection neurons receive sensory input
lambda_s = 0;           % proportion of interneurons receiving sensory input
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
%% Run sims
S = zeros(Nc*n*rho,n_ins);            % storage for input vectors
Pout = zeros(Nc*n*rho,n_ins);   % storage for output vectors    
Cout = zeros(Nc,n_ins);   % storage for output vectors    
converged = zeros(n_ins,1);

for loop = 1:n_ins
    %%%%%%%%%%%%%%%% input parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rand('state',loop);
    % random inputs to all projection units...
    flag = [flag 'p'];
    %S = exprnd(0.05,Nc*n,1); % most get small, some get large
    Sall = rand(Nc*n,1);    % must spec input to all cells
    
    %%%%%%%%%%%%%%% RUN FULL MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % sets seed, so should build same model every time, just changes input...
    [clust_out,clust_act,Proj_units,steps_elapsed,Clust_con,S_clust,proj_out,samp,t_samp,this_model] =...
        discrete_cluster1(Nc,n,Pl,Pp,rho_s,lambda_s,rho,rho_i,lambda_i,Pc,con,max_steps,theta,Sall,seed,flag,[],[],W);
    
    %converged?
    converged(loop) = real(steps_elapsed < max_steps);
    
    % get projection unit indices
    [r p_per_cluster] = size(Proj_units);   
    temp_p_units = zeros(Nc,p_per_cluster);
    for loop1 = 1:Nc
        temp_p_units(loop1,:) = Proj_units(loop1,:) + (loop1-1) * n;
    end
    temp_p_units = temp_p_units';
    idx_p_units = sort(temp_p_units(:));

    % get interneuron indices
    %idxCells = 1:n*Nc;
    %idx_i_units = setdiff(idxCells,idx_p_units)';

    S(:,loop) = Sall(idx_p_units);  % vector of all inputs to projection neurons
    Pout(:,loop) = proj_out(:);     % vector of all projection neuron outputs
    Cout(:,loop) = sum(proj_out)';  % vectors of all cluster outputs
end

%% compute "distances" between inputs

% Eucledian distance between vectors
in_dist = pdist(S'); % Eucledian distance 
Cout_dist = pdist(Cout');
Pout_dist = pdist(Pout');
[BdistC,BINT,R,RINT,STATSdistC] = regress(Cout_dist',[ones(numel(in_dist),1) in_dist']);
[BdistP,BINT,R,RINT,STATSdistP] = regress(Pout_dist',[ones(numel(in_dist),1) in_dist']);

% angle between vectors
in_cos = pdist(S','cosine');
Cout_cos = pdist(Cout','cosine');
Pout_cos = pdist(Pout','cosine');
[BcosC,BINT,R,RINT,STATScosC] = regress(Cout_cos',[ones(numel(in_cos),1) in_cos']);
[BcosP,BINT,R,RINT,STATScosP] = regress(Pout_cos',[ones(numel(in_cos),1) in_cos']);

% correlation between vectors
in_correlation = pdist(S','correlation');
Cout_correlation = pdist(Cout','correlation');
Pout_correlation = pdist(Pout','correlation');
[BcorrelationC,BINT,R,RINT,STATScorrelationC] = regress(Cout_correlation',[ones(numel(in_correlation),1) in_correlation']);
[BcorrelationP,BINT,R,RINT,STATScorrelationP] = regress(Pout_correlation',[ones(numel(in_correlation),1) in_correlation']);

% plot distances....
figure(1); clf
subplot(311),plot(in_dist,Cout_dist,'.'); hold on; plot(in_dist,BdistC(1) + BdistC(2)*in_dist,'k')
xlabel('Input distance'); ylabel('Cluster output distance'); title(['r^2 = ' num2str(STATSdistC(1)) '; p = ' num2str(STATSdistC(3))]); 
subplot(312),plot(in_cos,Cout_cos,'.'); hold on; plot(in_cos,BcosC(1) + BcosC(2)*in_cos,'k')
xlabel('Input angle'); ylabel('Cluster output angle'); title(['r^2 = ' num2str(STATScosC(1)) '; p = ' num2str(STATScosC(3))]); 
subplot(313),plot(in_correlation,Cout_correlation,'.'); hold on; plot(in_correlation,BcorrelationC(1) + BcorrelationC(2)*in_correlation,'k')
xlabel('Input correlation'); ylabel('Cluster output correlation'); title(['r^2 = ' num2str(STATScorrelationC(1)) '; p = ' num2str(STATScorrelationC(3))]); 


figure(2); clf
subplot(311),plot(in_dist,Pout_dist,'.'); hold on; plot(in_dist,BdistP(1) + BdistP(2)*in_dist,'k')
xlabel('Input distance'); ylabel('Projection neuron output distance'); title(['r^2 = ' num2str(STATSdistP(1)) '; p = ' num2str(STATSdistP(3))]); 
subplot(312),plot(in_cos,Pout_cos,'.'); hold on; plot(in_cos,BcosP(1) + BcosP(2)*in_cos,'k')
xlabel('Input angle'); ylabel('Projection neuron output angle'); title(['r^2 = ' num2str(STATScosP(1)) '; p = ' num2str(STATScosP(3))]); 
subplot(313),plot(in_correlation,Pout_correlation,'.'); hold on; plot(in_correlation,BcorrelationP(1) + BcorrelationP(2)*in_correlation,'k')
xlabel('Input correlation'); ylabel('Projection neuron output correlation'); title(['r^2 = ' num2str(STATScorrelationP(1)) '; p = ' num2str(STATScorrelationP(3))]); 


save proj_input_IO_results




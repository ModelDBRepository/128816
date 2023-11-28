function [clust_out,clust_act,Proj_units,steps_elapsed,Clust_con,S_clust,proj_out,varargout] = discrete_cluster1(N_clust,N,P_LC,P_PC,Proj_S,Other_S,Proj,Proj_I,Other_I,r,convergence,max_steps,thresh,sensory_input,seed,varargin)

% DISCRETE_CLUSTER1 simulate discrete sensory input to RF cluster model version 1
%
% [A,B,C,D,E,F,G] = DISCRETE_CLUSTER1(NC,NN,PL,PP,S,SO,P,PI,OI,R,CON,MAX,THETA,SENS,SEED)
%   NC clusters, NN neurons per cluster, 
%   PL probability of local connections; PP probability of contacts per long range connecton to cluster
%   S proportion of projection neurons which receive sensor input (usually 1), SO proportion of interneurons which
%   receive sensory input, P proportion of projection units,
%   PI proportion of projection neurons which are inhibitory, OI proportion of interneurons which are inhibitory,
%   R power law exponent for long-range connections (but see below), CON convergence criteria, MAX maxmimum number of time-steps allowed,
%   THETA threshold for leaky-integrator units, SENS array of sensory
%   input, SEED random number generator seed (set to [] to get new model
%   on each call)
%
%   NOTE: intra- and inter- cluster connections (as specified by PL and PP
%   respectively) are selected from a uniformly random distribution.
%
%   returns A the output array of the entire model, B the activation array if the model, C the indices of the cluster projection units, D the number of steps elapsed,
%   E the cluster connection matrix, F the matrix of units receiving sensory inputs, G is a matrix of projection unit outputs
%   where each column is a cluster.
%   
%   Add optional outputs: H returns the matrix of sample points of outputs
%   from the network, I returns the time-stamps of those outputs, J returns a structure describing structural features of the model, 
%
%   DISCRETE_CLUSTER1(...,FLAG,PA,PO,W) where FLAG contains any of
%           's' - uses spatially-uniform distribution of long-range
%           connections rather than distance-dependent. Parameter R becomes
%           probability of connections and therefore must be in range
%           {0,1}.
%
%           'r' - sets the weights to Gamma-distributed random values (centered around
%           +/- W); allows the model to operate across its entire dynamic
%           range (default weight is 1)
%           
%           'f' - sets weights to constant of W (default weight is 1)
%
%           'p' - allows input array SENS to be complete sensory-receiving cell pattern, rather than specifying
%           by cluster;   
%
%           'i' - scales inhibitory weights by the ratio between
%           excitatory:inhibitory connections across the whole model
%
%           'o' - solves the model using the numerical ODE version of the
%           neural net solver rather than discrete-time exact solution. The
%           numerical version is *much* slower, but useful for verifying
%           complex dynamics are not an artifact of the discrete-time
%           assumptions.
%       
%   and where PA and PO are the activation and output arrays to initialise 
%   the model into a prior state - e.g. the state acheived after the last set of inputs. Note that
%   these cannot be used with the ODE solver version.
%   Set FLAG = '' to omit if required. Set PA, PO = [] to omit (if only W
%   required)
%
%   Mark Humphries 2/12/2005

uniform = 0;
rand_w = 0;
pattern = 0;
scale_i = 0;
exact_solver = 1;
fixed_w = 1;

if nargin >= 16
    if findstr(varargin{1},'s') uniform = 1; end
    if findstr(varargin{1},'r') rand_w = 1; end
    if findstr(varargin{1},'p') pattern = 1; end
    if findstr(varargin{1},'f') 
        if nargin < 19
            error('Fixed weight requested but not supplied as final argument - see help for details');
        else
            fixed_w = varargin{4}; 
        end
    end
    if findstr(varargin{1},'i') scale_i = 1; end
    if findstr(varargin{1},'o') exact_solver = 0; end
end

if N_clust < 1
    error('Must specify at least 2 clusters')
elseif ~pattern & N_clust ~= length(sensory_input)
    error('Sensory input array must have same number of elements as there are specified clusters');
end

%% set random seed for replicable results
if ~isempty(seed)
    rand('state',seed);
    randn('state',seed);
end

% number of units total
N_units = N_clust * N;

%% simulation parameters
k = 50;        % tau ~ 2ms
% k = 5;       % tau ~ 200 ms 
slope = 1;
delta_t = 0.0001;       % NOTE: dt << tau !!
decay = exp(-k * delta_t);
%convergence = 0.0001;
%num_runs = 100;
%thresh = -0.1;

%%%%%%% the high-level processing unit is termed a cluster %%%%%%%%%%%%%
%%%%% cluster parameters - number of neurons and input types
num_Proj = ceil(Proj * N);                  % number of projection units
num_Other = N - num_Proj;                   % number of other units
num_Proj_S = ceil(num_Proj * Proj_S);       % number of projection units receiving sensory input
num_Other_S = ceil(num_Other * Other_S);    % number of other units receiving sensory input
num_Proj_I = ceil(num_Proj * Proj_I);       % number of inhibitory projection units
num_Other_I = ceil(num_Other * Other_I);    % number of inhibitory other units

Clust_con = zeros(N_units);   % connection matrix - rows from, columns to

S_clust = zeros(N_clust,N);
Clust_sign = ones(N_clust,N);
Proj_units = zeros(N_clust,num_Proj);

%%% generate clusters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Clust = zeros(N_clust);
I_to_I = zeros(N_clust,1);
I_to_P = zeros(N_clust,1);
P_to_P = zeros(N_clust);
P_to_I = zeros(N_clust);

for clust = 1:N_clust   
    %% determine projection units
    sequence = randperm(N);
    projection = sequence(1:num_Proj);              %% array of projection unit indices 
    Proj_units(clust,:) = projection;
    
    Clust_sign(clust,projection(1:num_Proj_I)) = -1;    % assign inhibitory projection units
        
    %% detemine other inhibitory units
    others = sequence(num_Proj+1:end);                  %% array of other unit indices
    inhibitory = others(1:num_Other_I);
    Clust_sign(clust,inhibitory) = -1;
    
    %% determine sensory input units
    % projection
    sens = randperm(num_Proj);
    proj_sens = projection(sens(1:num_Proj_S));
    S_clust(clust,proj_sens) = 1;
    % other
    sens = randperm(num_Other);
    other_sens = others(sens(1:num_Other_S));
    S_clust(clust,other_sens) = 1;

    
    %%%%%%%%%% inter-cluster connectivity %%%%%%%%%%%%%%%%%%%
    if ~uniform
        %% use power law to define which clusters the current cluster units connect to
        disp('Distance-dependent model');
        for i = 1:num_Proj 
           for j = 1:N_clust  
                if j ~= clust                                   %% cannot connect long-range to self!!
                    prob_connect = abs(clust - j) .^ -r;         %% note: using this power law, a cluster will always contact its immediate neighbours
                    if rand < prob_connect
                        number_units = sum(rand(N,1) <= P_PC);         %% number of units in target cluster contacted by projection unit 0 <= number_units <= N   
                        if number_units > 0
                            Clust(clust,j) = Clust(clust,j) + 1; % generate cluster connection matrix for debugging
                        end
                        sequence = randperm(N);
                        row = projection(i)+(clust-1)*N;
                        col = sequence(1:number_units)+(j-1)*N;
                      
                        Clust_con(row,col) = Clust_sign(clust,projection(i));
                    end
                end
           end
        end
    else
        %% use binomial (spatially-uniform) distribution to define which clusters the current cluster units connect to
        disp('Spatially-uniform model');
        if r > 1 error('Probability of long-range connection must be in range {0,1}.'); end
            
        for i = 1:num_Proj 
           for j = 1:N_clust  
                if j ~= clust                                           %% cannot connect long-range to self!!
                    if rand < r                                         %% uniformly random distribution of cluster targets
                        number_units = sum(rand(N,1) <= P_PC);          %% number of units in target cluster contacted by projection unit 0 <= number_units <= N   
                        if number_units > 0
                            Clust(clust,j) = Clust(clust,j) + 1; % generate cluster connection matrix for debugging
                        end
                        sequence = randperm(N);
                        row = projection(i)+(clust-1)*N;
                        col = sequence(1:number_units)+(j-1)*N;
                       
                        Clust_con(row,col) = Clust_sign(clust,projection(i));
                    end
                end
           end
        end
    end
    
    %% determine intra-cluster connections: only interneurons have them
    for i = 1:num_Other
        local_con_probs = rand(N,1);                        %% probabilites of connecting to each unit in cluster 
        cons = find(local_con_probs <= P_LC);
        
        % determine what it's connecting to
        for loop = 1:length(cons)
            if any(projection == cons(loop))
                I_to_P(clust) = I_to_P(clust) + 1;
            else
                I_to_I(clust) = I_to_I(clust) + 1;
            end
        end
        
        row = others(i)+(clust-1)*N;
        col = cons + (clust-1)*N;
        Clust_con(row,col) = Clust_sign(clust,others(i)); 
        Clust_con(row,row) = 0;                             %% remove any recurrent connections!
    end
end

% determine linear index of projection units
idx_p_units = zeros(N_clust,num_Proj);
for loop1 = 1:N_clust
    idx_p_units(loop1,:) = Proj_units(loop1,:) + (loop1-1) * N;
end
idx_p_units = idx_p_units';
idx_p_units = sort(idx_p_units(:));

% find inter-cluster projection structure
for loop1 = 1:N_clust
   i_start = 1 + (N*(loop1-1));
   i_end = loop1*N;
   this_clust = idx_p_units(idx_p_units >= i_start & idx_p_units <= i_end); 
   this_clust = this_clust - N *(loop1-1); % shift back

%    this_idx = zeros(N,1);
%    this_idx(this_clust) = 1;
   
   for loop2 = 1:N_clust 
       if loop2 ~= loop1
           this_section = Clust_con(1+N*(loop2-1):N*loop2,i_start:i_end);
           this_section = sum(this_section);
           to_P = sum(this_section(this_clust));
           P_to_P(loop2,loop1) = to_P;
           P_to_I(loop2,loop1) = sum(this_section) - to_P;
       end
   end
end

% find neuron input balance per cluster and per P and I units
input_w = sum(Clust_con);
P_pos = zeros(N_clust,1);
P_neg = zeros(N_clust,1);
P_bal = zeros(N_clust,1);
I_pos = zeros(N_clust,1);
I_neg = zeros(N_clust,1);
I_bal = zeros(N_clust,1);

for loop = 1:N_clust
   i_start = 1 + (N*(loop-1));
   i_end = loop*N;
   this_clust_p = idx_p_units(idx_p_units >= i_start & idx_p_units <= i_end); 
   this_clust_i = i_start:i_end;
   this_clust_i(this_clust_p-N*(loop-1)) = [];
   
   P_pos(loop) = sum(input_w(this_clust_p) > 0);
   P_neg(loop) = sum(input_w(this_clust_p) < 0);
   P_bal(loop) = sum(input_w(this_clust_p) == 0);
   I_pos(loop) = sum(input_w(this_clust_i) > 0);
   I_neg(loop) = sum(input_w(this_clust_i) < 0);
   I_bal(loop) = sum(input_w(this_clust_i) == 0);

end

% generate structure of model structural values other than weight matrix
this_model = struct('Clust',Clust,'I_to_I',I_to_I,'I_to_P',I_to_P,...
    'P_to_P',P_to_P,'P_to_I',P_to_I,'P_pos',P_pos,'P_neg',P_neg,'P_bal',P_bal,...
    'I_pos',I_pos,'I_neg',I_neg,'I_bal',I_bal);

%%%%%%%% set weights %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% MUST SORT OUT THIS SECTION BEFORE SERIOUS STUDY OF MODEL
%%%%%%% %%%%%%%%%%%%%%%%
% set weights to random values if necessary
    pre_C_e = length(find(Clust_con > 0));
    pre_C_i = length(find(Clust_con < 0));

%if rand_w Clust_con = Clust_con .* rand(size(Clust_con)); end  % scales all weights
if rand_w Clust_con = Clust_con .* random('Gamma',10,0.02,N_units,N_units); % scales all weights
% set fixed weight....
else Clust_con = Clust_con .* fixed_w; end

C_e = length(find(Clust_con > 0));
C_i = length(find(Clust_con < 0));

% scale inhibitory weights
if scale_i
    Clust_con(Clust_con < 0) = Clust_con(Clust_con < 0) .* (C_e / C_i);
end

%keyboard
%Clust_con = Clust_con ./ sum(sum(abs(Clust_con)));
% Clust_con = zeros(N_units);

%%% look at model as lattice ring
% visualise_net(Clust_con,'undirected');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  run simulation      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% create sensory input matrix %%%
if ~pattern	
    [m n] = size(sensory_input);

	% replicate array
	if n > m
        temp_sens = sensory_input(ones(1,N),:);
	else
        temp_sens = sensory_input(:,ones(1,N));
	end
   
%     % convert to continuous array
% 	temp_sens = temp_sens(:);
% 	
% 	% and convert input unit matrix to continuous array
% 	temp_S_clust = S_clust';
% 	temp_S_clust = temp_S_clust(:);
	
	% therefore input array is 
	input = temp_sens .* S_clust;
    input = input';
    input = input(:);
else
    temp_S_clust = S_clust';
	temp_S_clust = temp_S_clust(:);
    input = sensory_input .* temp_S_clust;
end


%%%% create suitable arrays %%%%%
a_Thresh = thresh*ones(N_units,1);
a_Slope = slope*ones(N_units,1);
a_Decay = decay .*ones(N_units,1);
a_Tau = (1/k) .* ones(N_units,1);

%% solve network %%
s_points = -1;  % no samples
s_points = 1:1:max_steps;

% select solution method
if exact_solver
	if nargin >= 17 & ~isempty(varargin{2}) & ~isempty(varargin{3})
        [clust_out,steps_elapsed,clust_act,y] = LI_network(Clust_con,input,a_Thresh,a_Slope,s_points,max_steps,convergence,a_Decay,N_units,0,varargin{2},varargin{3});
	else
        [clust_out,steps_elapsed,clust_act,y] = LI_network(Clust_con,input,a_Thresh,a_Slope,s_points,max_steps,convergence,a_Decay,N_units,0);
        %[clust_out,steps_elapsed,clust_act,y] = LI_network_num(Clust_con,input,a_Thresh,a_Slope,s_points,a_Tau,max_steps,convergence,delta_t,N_units,0);
    end
	varargout{1} = y;
	[r c] = size(y);
	varargout{2} = s_points(1:c) * delta_t;
else
	disp('Using built-in ODE solver')
	tspan = [0 max_steps * delta_t];
	y0 = zeros(N_units,1);
	[act,t,out] = LI_network_ode(Clust_con,input,N_units,1/k .* ones(N_units,1),'ramp',tspan,y0);
	varargout{1} = out';
	varargout{2} = t;
    clust_out = out(end,:);
	clust_act = act(end,:);
	steps_elapsed = t(end) / delta_t;
end

%% output of most interest is from projection units....    
proj_out = reshape(clust_out(idx_p_units),num_Proj,N_clust);

%% also return all other structural values
varargout{3} = this_model;

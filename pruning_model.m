function [Clust_con,varargout] = pruning_model(Nc,n,rho,Pc,seed,L,T,model,flags,varargin)

% PRUNING_MODEL synaptic pruning RF anatomical model
%   PRUNING_MODEL(NC,N,RHO,Pc,SEED,L,T,M,FLAGS) where NC, N, Pc, and RHO are the parameters 
%   for the discrete cluster model, SEED is a random seed initialiser (set SEED=[] 
%   if not required), L is the proportion of neurons which randomly update, T is the
%   synaptic pruning threshold, M is the model type ('random','prop','inc') 
%   and FLAGS is an the argument setting the flags for
%   the DISCRETE_CLUSTER1 function (set FLAGS = [] to omit).
%
%   Returns the weight matrix - must be converted into graph form for
%   further topological analysis.
%
%   PRUNING_MODEL(...,TPl,TPp,'s') are optional arguments:
%   TPl and TPp set the target P(l) and P(p) values for the synaptic total to be 
%   pruned too. Default is P(l) = P(p) = 0.2, set both to [] to omit; 's'
%   forces the model to separately update excitatory and inhibitory units
%   to get both projection and inter-neuron totals close to expected values
%   (rather than overall synaptic total).
%
%
%   NOTE: 
%   (1) Pc is the either the probability of connection
%   (spatially-uniform - include flag 's') or exponent of 
%   the power-law (distance-dependent - default). In manuscript, these have
%   been set at Pc = 0.25 ('s') or Pc = 1 (default).
%   
%   (2) L is the proportion of units chosen to be updated. For
%   the 'prop' and 'inc' models, the units are chosen with probability proportional to
%   their total weighted input; for the 'random' model, the proportion L
%   are chosen at random. 
%
%   (3) For the 'inc' model, all units chosen are increased, all others are
%   decreased.....
%
%   Mark Humphries 15/7/2005

t_Pl = 0.2;     % default target levels of P(l) and P(p)
t_Pp = 0.2;
count_all = 1;

if nargin >= 10 & ~isempty(varargin{1}) t_Pl = varargin{1}; end
if nargin >= 11 & ~isempty(varargin{2}) t_Pp = varargin{2}; end
if nargin >= 12 & ~isempty(varargin{3}) count_all = 0; end

flags = ['r' flags];    % default flag to generate random weights 

%%%%%%%%%%%% define general structural model parameters %%%%%%%%%%%
% Inter-cluster
Pp = 0.9;           % probability of cluster-unit connection for long-range input 

% Intra-cluster
Pl = 0.9;           % probability of connection for each local interneuron
rho_s = 1;             % all projection neurons receive sensory input
lambda_s = 0;           % proportion of interneurons receiving sensory input
rho_i = 0;             % proportion of inhibitory projection neurons
lambda_i = 1;             % proportion of inhibitory interneurons

%%%%%%%%%%% parameters for FULL model   %%%%%%%%%%%%%%%%%%%%%%%%%
% simulation parameters
con = 1e-6;         % convergence criteria
max_steps = 1;
theta = 0;          % unit threshold
slope = 1;          % output slope
% input
S = zeros(Nc,1);   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% CREATE FULL MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[clust_out,clust_act,Proj_units,steps_elapsed,Clust_con,S_clust,proj_out,samp,t_samp,this_model] =...
    discrete_cluster1(Nc,n,Pl,Pp,rho_s,lambda_s,rho,rho_i,lambda_i,Pc,con,max_steps,theta,S,seed,flags);

clear clust_out clust_act steps_elapsed S_clust proj_out samp t_samp

%%%%%%%%%% CREATE PRUNED MODEL %%%%%%%%%%%%%%%%%%%%%%
n_p = round(rho * n);
n_i = n - n_p;

% find expected numbers of connection types
Ei = round(Nc * n_i * (n-1) * t_Pl);

if findstr(flags,'s')
    Ep = round(n_p * Nc * n * t_Pp * (Nc-1) * Pc); 
else
    Nq = 0;
    for loop = 1:Nc
      j = min(Nc - loop,(Nc-1)-(Nc-loop));
      k = max(Nc - loop,(Nc-1)-(Nc-loop));
      Nq = Nq + (2*(sum((1:j).^-Pc)) + sum((j+1:k).^-Pc));
    end 
    Ep = round(n_p * Nq * n * t_Pp);

end
TargetTotal = Ep + Ei;

%% initial synaptic totals and unit types
Syns = find(Clust_con ~= 0);
SynTotal = length(Syns);

% prune model!
counter = 0;

if count_all
    while SynTotal > TargetTotal
        counter = counter + 1;
        
        Remaining = SynTotal - TargetTotal        
        switch model
            case 'random'
                % randomly update - select random sub-set of synapses and add
                % random amount to each
                NumChange = round(SynTotal * L);
                List = randperm(NumChange);
                
                Old_w = Clust_con(Syns(List));
                Clust_con(Syns(List)) = Old_w + normrnd(0,0.025,NumChange,1);
                
                % any that change sign are automatically pruned...
                FlipSign = find((Clust_con(Syns(List)) > 0 & Old_w < 0) | (Clust_con(Syns(List)) < 0 & Old_w > 0));
                Clust_con(Syns(List(FlipSign))) = 0;
            case {'prop','inc'}
                % update whole unit's output with probability proportional 
                % to absolute total input weight - on basis that, according to
                % Hebbian-type learning:
                %
                % 'prop': greater total input would lead to greater chance of either
                % correct output (leading to LTP) or incorrect output (leading to LTD) 
                %
                % 'inc': greater total input signifies mostly correct responses
                % would occur and therefore be strengthend more, all others
                % decreased....
                
                % do excitatory
                input_totals = sum(Clust_con);
                output_totals = sum(Clust_con');
                total_input = sum(abs(input_totals));
                probabilities = input_totals ./ total_input; 
               
                % check probabilities
                update_prop = 0;
                List = [];
                n_units = Nc*n;
                units = 1:n_units;
                
                while update_prop < L
                    % get units to update by checking probabilities
                    update = find(rand(1,n_units) < probabilities); 
                    
                    % add corresponding units to list
                    List = [List units(update)];
                    
                    % remove from possible candidates
                    units(update) = [];
                    probabilities(update) = [];
                    
                    % reduce number of units to update
                    n_units = n_units - length(update);
                    update_prop = length(List) / (Nc*n);    
                end
                Old_w = Clust_con;
                change = zeros(Nc*n);
                change(:,List) = Clust_con(:,List) ~= 0;
                
                % shift weights a random amount - more likely to be active, but
                % also therefore likely to fire incorrectly and therefore be reduced in
                % efficacy
                
                % must multiply by connection matrix so that only existing
                % connections get changed!
                if findstr(model,'prop')
         
                    Clust_con = Old_w + normrnd(0,0.025,Nc*n,Nc*n) .* change;       
                else    
                    % or shift all in dominant direction - assume strengthening
                    % Gamma distribution here approximates RHS of normal dist
                    % used above
                    Clust_con = Clust_con + gamrnd(1,0.015,Nc*n,Nc*n) .* change .* repmat(sign(input_totals),Nc*n,1);
                    
                    % all others decrease - variable "units" contains all those
                    % not already updated...
                    change_down = zeros(Nc*n);
                    change_down(:,units) = Clust_con(:,units) ~= 0;
                    Clust_con = Clust_con - gamrnd(1,0.015,Nc*n) .* change_down .* repmat(sign(input_totals),Nc*n,1);
                    
                end
                
                % any that change sign are automatically pruned...
                FlipSign = find((Clust_con > 0 & Old_w < 0) | (Clust_con < 0 & Old_w > 0));
                Clust_con(FlipSign) = 0; 
                        
        end 
                    
        % prune synapses by threshold
        Clust_con(abs(Clust_con) < T) = 0;
        
        Syns = find(Clust_con ~= 0);
        SynTotal = length(Syns);
    end
        
else
    %%%% do separate running totals	
    % unit indices
	clusters = repmat((0:Nc-1)',1,n_p);
	IdxP = (Proj_units + (n .* clusters))';
	IdxP = IdxP(:);         % projection neuron indices
	nP = length(IdxP);
    
	IdxI = (1:n*Nc)';
	IdxI(IdxP) = [];        % interneuron indices
	nI = length(IdxI);
    
    ClustP = Clust_con(IdxP,:);     % outputs from each projection neuron
    ClustI = Clust_con(IdxI,:);
    
    ClustP = Clust_con(:,IdxP);     % inputs to each projection neuron
    ClustI = Clust_con(:,IdxI);

	tempP = Clust_con(IdxP,:);
    tempI = Clust_con(IdxI,:);
    SynP = find(tempP);
	SynI = find(tempI);
    SynPTotal = length(SynP);
    SynITotal = length(SynI);
    
    
    %% do separate running totals for projection and inter-neurons
	while SynPTotal > Ep | SynITotal> Ei    
        counter = counter + 1;
        
        P_remaining = SynPTotal - Ep
        I_remaining = SynITotal - Ei
        
        % alter distribution of update rates according to number of units
        % remaining, with the aim of them finishing together
        rp = (P_remaining*(P_remaining > 0)) / ((I_remaining*(I_remaining>0)) + P_remaining);
        ri = 1 - rp;
        Lp = L * rp;
        Li = L * ri;
        
        switch model
            case 'random'
                % randomly update - select random sub-set of synapses and add
                % random amount to each
                if P_remaining > 0
                    NumPChange = round(SynPTotal * Lp);
                else
                    NumPChange = 0;
                end
                if I_remaining > 0
                    NumIChange = round(SynITotal * Li);
                else
                    NumIChange = 0;
                end
                ListP = randperm(NumPChange);
                ListI = randperm(NumIChange);
                
                % keyboard
                % update P
                Old_w = ClustP(SynP(ListP));
                ClustP(SynP(ListP)) = Old_w + normrnd(0,0.025,NumPChange,1);
                
                % any that change sign are automatically pruned...
                FlipSign = find((ClustP(SynP(ListP)) > 0 & Old_w < 0) | (ClustP(SynP(ListP)) < 0 & Old_w > 0));
                ClustP(SynP(ListP(FlipSign))) = 0;
                                
                % update I
                Old_w = ClustI(SynI(ListI));
                ClustI(SynI(ListI)) = Old_w + normrnd(0,0.025,NumIChange,1);
                
                % any that change sign are automatically pruned...
                FlipSign = find((ClustI(SynI(ListI)) > 0 & Old_w < 0) | (ClustI(SynI(ListI)) < 0 & Old_w > 0));
                ClustI(SynI(ListI(FlipSign))) = 0;
                
                % put back into Clust_con
                %Clust_con(:,IdxP) = ClustP;
                %Clust_con(:,IdxI) = ClustI;
                
                Clust_con(IdxP,:) = ClustP;     % inputs to each projection neuron
                Clust_con(IdxI,:) = ClustI;

    
        case {'prop','inc'}
                % update whole unit's output with probability proportional 
                % to absolute total input weight - on basis that, according to
                % Hebbian-type learning:
                %
                % 'prop': greater total input would lead to greater chance of either
                % correct output (leading to LTP) or incorrect output (leading to LTD) 
                %
                % 'inc': greater total input signifies mostly correct responses
                % would occur and therefore be strengthened more, all others
                % decreased....
                
                input_totals = sum(Clust_con);
                output_totals = sum(Clust_con');
                total_input = sum(abs(input_totals));
                
                % keyboard
                P_probabilities = input_totals(IdxP) ./ total_input; 
                I_probabilities = input_totals(IdxI) ./ total_input; 

                %%%% do projection neurons
                if P_remaining > 0 
                    % check probabilities
                    update_prop = 0;
                    List = [];
                    nP = length(IdxP);
                    p_units = 1:nP;
                    
                    while update_prop < Lp
                        % get units to update by checking probabilities
                        update = find(rand(1,nP) < P_probabilities); 
                        
                        % add corresponding units to list
                        List = [List p_units(update)];
                        
                        % remove from possible candidates
                        p_units(update) = [];
                        P_probabilities(update) = [];
                        
                        % reduce number of units to update
                        nP = nP - length(update);
                        update_prop = length(List) / length(IdxP);    
                    end
                    
                    Old_w = ClustP;
                    change = zeros(Nc*n,length(IdxP));
                    change(:,List) = ClustP(:,List) ~= 0;
                    
                    % shift weights a random amount - more likely to be active, but
                    % also therefore likely to fire incorrectly and therefore be reduced in
                    % efficacy
                    
                    % must multiply by connection matrix so that only existing
                    % connections get changed!
                    if findstr(model,'prop')
             
                        ClustP = Old_w + normrnd(0,0.025,Nc*n,length(IdxP)) .* change;       
                    else    
                        % or shift all in dominant direction - assume strengthening
                        % Gamma distribution here approximates RHS of normal dist
                        % used above
                        
                        ClustP = Old_w + gamrnd(1,0.015,Nc*n,length(IdxP)) .* change .* repmat(sign(input_totals(IdxP)),Nc*n,1);
                        
                        % all others decrease - variable "p_units" contains all those
                        % not already updated...
                        change_down = zeros(Nc*n,length(IdxP));
                        change_down(:,p_units) = Clust_con(:,p_units) ~= 0;
                        ClustP = ClustP - gamrnd(1,0.015,Nc*n,length(IdxP)) .* change_down .* repmat(sign(input_totals(IdxP)),Nc*n,1);
                        
                    end
                    
                    % any that change sign are automatically pruned...
                    FlipSign = find((ClustP > 0 & Old_w < 0) | (ClustP < 0 & Old_w > 0));
                    ClustP(FlipSign) = 0; 
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% do inter-neurons
                %%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if I_remaining > 0   
                    % check probabilities
                    update_prop = 0;
                    List = [];
                    nI = length(IdxI);
                    i_units = 1:nI;
                    
                    while update_prop < Li
                        % get units to update by checking probabilities
                        update = find(rand(1,nI) < I_probabilities); 
                        
                        % add corresponding units to list
                        List = [List i_units(update)];
                        
                        % remove from possible candidates
                        i_units(update) = [];
                        I_probabilities(update) = [];
                        
                        % reduce number of units to update
                        nI = nI - length(update);
                        update_prop = length(List) / length(IdxI);    
                    end
                    % keyboard
                    
                    Old_w = ClustI;
                    change = zeros(Nc*n,length(IdxI));
                    change(:,List) = ClustI(:,List) ~= 0;
                    
                    % shift weights a random amount - more likely to be active, but
                    % also therefore likely to fire incorrectly and therefore be reduced in
                    % efficacy
                    
                    % must multiply by connection matrix so that only existing
                    % connections get changed!
                    if findstr(model,'prop')
                        % keyboard
                        ClustI = Old_w + normrnd(0,0.025,Nc*n,length(IdxI)) .* change;       
                        
                    else    
                        % or shift all in dominant direction - assume strengthening
                        % Gamma distribution here approximates RHS of normal dist
                        % used above
                        ClustI = Old_w + gamrnd(1,0.015,Nc*n,length(IdxI)) .* change .* repmat(sign(input_totals(IdxI)),Nc*n,1);
                        
                        % all others decrease - variable "i_units" contains all those
                        % not already updated...
                        change_down = zeros(Nc*n,length(IdxI));
                        change_down(:,i_units) = Clust_con(:,i_units) ~= 0;
                        ClustI = ClustI - gamrnd(1,0.015,Nc*n,length(IdxI)) .* change_down .* repmat(sign(input_totals(IdxI)),Nc*n,1);
                        
                    end
                    
                    % any that change sign are automatically pruned...
                    FlipSign = find((ClustI > 0 & Old_w < 0) | (ClustI < 0 & Old_w > 0));
                    ClustI(FlipSign) = 0; 
                end
                    
                % put back into Clust_con
                Clust_con(:,IdxP) = ClustP;
                Clust_con(:,IdxI) = ClustI;
                
        end 
                    
        % prune synapses by threshold
        Clust_con(abs(Clust_con) < T) = 0;
        
        % generate new input sub-matrices
%         ClustP = Clust_con(IdxP,:);     % outputs from each projection neuron
%         ClustI = Clust_con(IdxI,:);

        ClustP = Clust_con(:,IdxP);     % inputs to each projection neuron
        ClustI = Clust_con(:,IdxI);     % inputs to each interneuron
        
        % do synaptic counts ON OUTPUTS
        tempP = Clust_con(IdxP,:);
        tempI = Clust_con(IdxI,:);
		SynP = find(tempP);
		SynI = find(tempI);
        SynPTotal = length(SynP);
        SynITotal = length(SynI);
		        
    end
end
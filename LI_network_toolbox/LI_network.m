function [out,steps,act,sample_out] = LI_network(W,I,theta,slope,sample_points,max_steps,convergence,decay,N,pulse,varargin)

% LI_NETWORK simulates network of leaky integrator units (MEX function)
%
%   [OUT,STEPS,ACT,SAMP] = LI_NETWORK(W,I,THETA,SLOPE,S,M,CONVERGENCE,DECAY,N,PULSE) simulates a network of leaky integrator units with rectified outputs
%   (i.e. unit output function is a ramp of slope M).
%   The connectivity matrix is specified by W (an N*N matrix). External input to the units is specifed by array I, which
%   must be of length N. Values of each element n in W and I are limited to: 0 <= I(n), W(n) <= 1. The threshold and slope value of 
%   each unit's ramp function is specified in arrays THETA and SLOPE, respectively, both of which must be of length N. 
%   Simulation runs until all units' activations have changed less than CONVERGENCE for consecutive time-steps, 
%   or until M timesteps have elapsed, whichever is sooner. 
%
%   The unit's decay term DECAY is an array of [exp(-K * DT)] where K is
%   the reciprocal of the time constant and DT is the time-step of the simulation:
%   the array should be of length N.
%
%   For continuous external input to the units set PULSE = 0; for a pulsed input on the first step of the simulation, set PULSE = 1;
%   
%   S is an array of time-step sample points at which to take a snapshot of the network's output. If no outputs other than the final
%   state required, set S < 0;
%   
%   The function returns the final unit output (OUT) and activation (ACT) arrays, the number of time-steps elapsed (STEPS),
%   and the matrix of output snapshots (SAMP)
%
%   LI_NETWORK(...,PA,PO) where PA and PO are arrays of unit outputs and
%   activations respectively (as returned in ACT and OUT above) which are
%   used to initialise the network as it was at the end of the previous
%   run.
%
%   Mark Humphries 31/1/2005 (see accompanying C file for last MEX file revision) 

% do error-checking in MatLab
[M N] = size(W);
if M ~= N
    error('Weight matrix must be square');
elseif length(theta) ~= N
    error(['Threshold array must have ' num2str(N) ' elements']);
elseif length(slope) ~= N
    error(['Slope array must have ' num2str(N) ' elements']);
elseif length(decay) ~= N
    error(['Decay constants array must have ' num2str(N) ' elements']);
end

% do pre-processing in MatLab
n_samples = length(sample_points);

% initialise activation and outputs
if nargin > 10
    p_o = varargin{1};
    p_a = varargin{2};
else
    % initialise to zero
    p_o = zeros(N,1);
    p_a = zeros(N,1);
end

% run network
% NOTE: when changes are made to arguments passed to C-file function, just
% modify here so that all parent programs (which call this) need minimum
% modification
[out,steps,act,sample_out] = LI_network_C(W,I,theta,slope,sample_points,decay,max_steps,convergence,N,pulse,n_samples,p_o,p_a);

% reshape and tidy up sample output array
sample_out = reshape(sample_out,N,n_samples);

idx = find(sample_points > steps-1);
if ~isempty(idx)
    sample_out(:,idx) = [];     % remove all points after convergence
end
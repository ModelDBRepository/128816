function [y,t,out] = LI_network_ode(W,I,N,tau,out_type,tspan,y0,varargin)

% LI_NETWORK_ODE leaky integrator network ODE system
%   [ACT,T,OUT] = LI_NETWORK(W,I,N,TM,O,TSPAN,Y0) numerically solves a network of leaky integrator units. 
%   The connectivity matrix is specified by W (an N*N matrix). External input to the units is specifed by array I, which
%   must be of length N. Values of each element n in W and I are limited
%   to: 0 <= I(n), W(n) <= 1. Membrane time constant TM a vector of N values in seconds. The output function O is one of {'ramp','tanh'}. 
%   TSPAN and Y0 are the standard ODE solver variables specifying time
%   period of solution and initial conditions - note that Y0 must be a
%   length N vector.
%
%   The matrix of solution values ACT, time-stamps T, and output values OUT are returned
%
%   LI_NETWORK(...,M,THETA) optionally specifies the slope and threshold of the ramp
%   output function; these must both be vectors of length N.
%
%   Mark Humphries 11/2/2005

if nargin > 7
    m = varargin{1};
    theta = varargin{2};
else
    % defaults for ramp output
    m = ones(N,1);
    theta = zeros(N,1);
end
    
[M N] = size(W);
if M ~= N
    error('Weight matrix must be square');
elseif length(tau) ~= N
    error(['Membrane time constant array must have ' num2str(N) ' elements']);
elseif length(y0) ~= N
    error(['Initial condition array must have ' num2str(N) ' elements']);
elseif length(theta) ~= N
    error(['Threshold array must have ' num2str(N) ' elements']);
elseif length(m) ~= N
    error(['Slope array must have ' num2str(N) ' elements']);
end



% build ODE system
ode_system = [];

for loop = 1:N
    diff_eq = ['(-y(' num2str(loop) ')']; 
    % get inputs
    inputs = find(W(:,loop) ~= 0);
    for loop2 = 1:length(inputs)
        diff_eq = [diff_eq '+' num2str(W(inputs(loop2),loop)) '*output(y(' num2str(inputs(loop2)) '),''' out_type ''', ' num2str(m(loop)) ', ' num2str(theta(loop)) ')'];     
    end
    diff_eq = [diff_eq '+' num2str(I(loop)) ') / ' num2str(tau(loop))];
    
    ode_system = strvcat(ode_system,diff_eq);
end

% output direct version
% for loop = 1:N
%     diff_eq = ['output(-y(' num2str(loop) ')']; % add zero to make loop below easier
%     % get inputs
%     inputs = find(W(:,loop) ~= 0);
%     for loop2 = 1:length(inputs)
%         diff_eq = [diff_eq '+' num2str(W(inputs(loop2),loop)) '*y(' num2str(inputs(loop2)) ')'];     
%     end
%     diff_eq = [diff_eq '+' num2str(I(loop)) ',''' out_type ''', ' num2str(m(loop)) ', ' num2str(theta(loop)) ') / ' num2str(tau(loop))];
%     
%     ode_system = strvcat(ode_system,diff_eq);
% end

[t,y] = ode45(@LI_ode,tspan,y0,[],ode_system,N);


% for LI units, returned y is activation - compute outputs
out = zeros(size(y));
for loop = 1:length(t)
    %keyboard
    out(loop,:) = output(y(loop,:)',out_type,m,theta)';
end

function dydt = LI_ode(t,y,ode_system,N)

% LI_ODE ODE function for LI network
%   LI_ODE(T,Y,S,O) where T, Y are the standard ODE function inputs, S is
%   a string statements describing the ODE system which has to be executed
%   using eval().

dydt = zeros(N,1);
for loop = 1:N
    dydt(loop) = eval(ode_system(loop,:)); 
end

% output function
function y = output(x,out_type,m,theta)

switch out_type
    case 'ramp'
        y = x;
  %      x = m.*(x-theta);
        y(y<theta) = 0;
        y(y>1./m + theta) = 1;
  %      y(y >= theta & y <= 1./m + theta) = x(y >= theta & y <= 1./m + theta);
    case 'tanh'
        % always includes Heaviside
        y = tanh(x .* (x > 0));
    otherwise
        error('No output type selected')
end
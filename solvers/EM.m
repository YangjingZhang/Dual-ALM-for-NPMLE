%%%%%% EM for solving the NPMLE with unknown supports
%%%%%% input: observations X \in R^{n*d}
%%%%%%        Sigma: n numbers of diagonal covaraince matrices \in R^{n*d}
%%%%%%        m: number of grid points
%%%%%%        option.grid_initial: initial supports

function [x,supps,hist,k] = EM(X,SIGMA,m,options)

fprintf('\n')
disp('----------------- EM algorithm--------------------')
maxiter = 100;
stoptol = 1e-4;
[n,d] = size(X);
%% uniform mixture proportions
x = 1/m*ones(m,1);
if m < n
    supps = X(randperm(n,m),:);
elseif m == n
    supps = X;
end
if exist('options','var')
    if isfield(options,'supps_initial');    supps = options.supps_initial;    end
    if isfield(options,'stoptol');    stoptol = options.stoptol;    end
end
Sigma = reshape(SIGMA,d,n)';
inv_Sigma = 1.\Sigma;
%% inv(Sigma)*X
inv_SigmaX = inv_Sigma.*X;
%% PEM
for k = 1:maxiter
    %% L
    L = likelihood_matrix(X,supps,SIGMA);
    %% E step
    Lx = L*x;
    gamma_hat = L.*(ones(n,1)*x')./(Lx*ones(1,m));  
    %% M step
    supps = (gamma_hat'*inv_SigmaX)./(gamma_hat'*inv_Sigma);
    x = sum(gamma_hat)'/n;
    obj = sum(log(Lx))/n;
    hist.obj(k) = obj;
    fprintf('iter = %3d, log-likelihood = %5.8e \n', k, obj)
    if k > 1 && obj - obj_old  < stoptol
        break;
    end
    obj_old = obj;
end

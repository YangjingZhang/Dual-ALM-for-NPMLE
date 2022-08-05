%%%%%% PEM for solving the NPMLE with unknown supports
%%%%%% input: observations X \in R^{n*d}
%%%%%%        Sigma: n numbers of diagonal covaraince matrices \in R^{n*d}
%%%%%%        m: number of grid points
%%%%%%        option.grid_initial: initial supports

function [x,supps,hist,k] = PEM(X,SIGMA,m,options)

fprintf('\n')
disp(' ----------------- partial EM algorithm--------------------')
maxiter = 100;
stoptol = 1e-4;
printyes = 0;
[n,d] = size(X);
if m < n
    supps = X(randperm(n,m),:);
elseif m == n
    supps = X;
end
if exist('options','var')
    if isfield(options,'supps_initial');    supps = options.supps_initial;    end
    if isfield(options,'stoptol');    stoptol = options.stoptol;    end
    if isfield(options,'printyes');    printyes = options.printyes;    end
end
options.scaleL = 1;
options.approxL = 0;
options.printyes = printyes;
options.init_opt = 1;
options.stoptol = stoptol;
Sigma= reshape(SIGMA,d,n)';
inv_Sigma = 1./Sigma;
%% inv(Sigma)*X
inv_SigmaX = inv_Sigma.*X;
%% PEM
if isfield(options,'rowmax')
    options = rmfield(options,'rowmax');
end
L = likelihood_matrix(X,supps,SIGMA);
for k = 1:maxiter
    %% solve x
    [obj,x,~,~,~,~,~] = DualALM(L,options);   
    fprintf('iter = %3d, log-likelihood = %5.8e \n', k, -obj(1))
    if k > 1 
        options.init_opt = x;
        if obj_old(1) - obj(1) < stoptol
           break;
        end
    end
    hist.obj(k) = -obj(1);
    %% estimate supports
    posind = (x>0);
    sumposind = sum(posind);
    xtmp = x(posind);
    Ltmp = L(:,posind);
    Lx = Ltmp*xtmp; 
    gamma_hat = Ltmp.*(repmat(xtmp',n,1))./(repmat(Lx,1,sumposind));
    supps_update = (gamma_hat'*inv_SigmaX)./(gamma_hat'*inv_Sigma);
    supps(posind,:) = supps_update;
    if sumposind < m/3
        L(:,posind) = likelihood_matrix(X,supps_update,SIGMA);
    else
        L = likelihood_matrix(X,supps,SIGMA);
    end
    obj_old = obj;
end

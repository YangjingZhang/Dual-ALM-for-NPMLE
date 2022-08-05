%% ALM for solving the dual problem of MLE of mixture proportions
%%    -min    (-1/n) \sum_{j=1}^n  log(u_j)
%%     s.t.   (1/n)L'v <= e                              (D)
%%            (1/alpha)(u - v) = 0
%% L is a nonnegative n*m matrix, e denotes the vector of all ones
%% alpha > 0 is a user-defined scaling parameter
%%
%%     min    (-1/n) \sum_{j=1}^n  log(y_j) - e'x + 1
%%     s.t.   (1/n)Lx = (1/n)y                                     (P)
%%            x >= 0
%%
%% Input parameters:
%% L: a nonnegative n*m matrix
%% options: optional input parameter containing several fields
%% options.stoptol is a tolerance parameter
%% options.printyes is a flag, =1 print details; =0 not print details
%% options.maxiter sets an upper bound on the number of iterations
%% options.alpha is the scaling parameter(by default, alpha = n)
%% Output parameters:
%% obj: primal and dual objective function values
%% x,y: primal variables
%% u,v: dual variables
%% info: information containing several fields
%% runhist: running history during the iterations, containing several fields
%%
%% Yangjing Zhang 07 Jan 2021
function [obj,x,y,u,v,info,runhist] = DualALM(L,options)
%% parameters
stoptol = 1e-6;   % error tolerance
stopop = 1;       % stopping option
printyes = 1;     % print details
maxiter = 100;    % maximum number of iterations
sigma = 100;      % penalty parameter
scaleL = 1;       % scale rows of L
approxL = 0;      % low rank approximation of L
approxRank = 30;
init_opt = 0;
if exist('options','var')
    if isfield(options,'stoptol');     stoptol = options.stoptol;       end
    if isfield(options,'stopop');      stopop = options.stopop;         end
    if isfield(options,'printyes');    printyes = options.printyes;     end
    if isfield(options,'maxiter');     maxiter = options.maxiter;       end
    if isfield(options,'sigma');       sigma = options.sigma;           end
    if isfield(options,'scaleL');      scaleL = options.scaleL;         end
    if isfield(options,'approxL');     approxL = options.approxL;       end
    if isfield(options,'approxRank');  approxRank = options.approxRank; end
	if isfield(options,'init_opt');    init_opt = options.init_opt;     end
end
if printyes
    fprintf('\n*************************************************************************************');
    fprintf('\n ALM for the dual problem');
    fprintf('\n*************************************************************************************');
end
tstart = clock;
[n,m] = size(L);
%% scale L such that the maximal value of each row is 1
if scaleL
    if exist('options','var') && isfield(options,'rowmax')
        s = 1./options.rowmax;
    else
        s = 1./max(L,[],2);
    end
    if printyes
        fprintf('\n max/min scale = %3.1e/%3.1e',max(1./s),min(1./s));
    end
    if (n > 1e6)
       for i = 1:n
           L(i,:) = s(i)*L(i,:);
       end
    else
       L = s.*L; 
    end    
else
    s = ones(n,1);
end
LL.matrix = L;
%% low rank approximation of L
if approxL
    approxRank = ceil(approxRank);
    approxSucceed = 0;
    t1 = clock;
    [U,S,V] = svds(L,approxRank,'largest');
    t2 = etime(clock,t1);
    if printyes
        fprintf('\n approximate rank = %d',approxRank);
        fprintf('\n partial svd(L) = %3.1f seconds',t2);
        fprintf('\n eigenvalues <= %2.2e were truncated ',S(approxRank,approxRank));
        fprintf('\n ----------');
    end
    if S(approxRank,approxRank) <= min(10*stoptol,1e-4)
        ii = find(diag(S) < min(10*stoptol,1e-4),1);
        if ~isempty(ii)
            U = U(:,1:ii);
            S = S(1:ii,1:ii);
            V = V(:,1:ii);
            approxRank = ii;
            if printyes
                fprintf('\n approximate rank = %d (further truncated)',approxRank);
                fprintf('\n eigenvalues <= %2.2e were truncated ',S(approxRank,approxRank));
                fprintf('\n ----------');
            end
        end
        U = U*S;
        LL.U = U;
        LL.V = V;
        LL.times = @(x) (U*(V'*x));
        LL.trans = @(y) (V*(U'*y));
        approxSucceed = 1;
    end
    if ~approxSucceed && printyes
        fprintf('\n numerical rank of L > %d, set approxL = 0',approxRank);
        fprintf('\n ----------');
        approxL = 0;
        LL.times = @(x) (L*x);
        LL.trans = @(y) (y'*L)';
    end
else
    LL.times = @(x) (L*x);
    LL.trans = @(y) (y'*L)';
end
%% initialization
if init_opt == 0
    xnew = ones(m,1)/m;
    ynew = sum(L,2)/m;
    unew = 1./ynew;
    vnew = unew;
else
    xnew = 0.5*sigma*ones(m,1);
    ynew = sum(L,2)/m;
    unew = 1./ynew;
    vnew = zeros(n,1);
end
%%
Lx = LL.times(xnew);
tmp = LL.trans(1./(Lx))/n - 1;
pkkt = norm(xnew - max(xnew + tmp,0)); 
if (pkkt < stoptol)
    x = xnew;
    obj = sum(x) + sum(log(s) - log(Lx))/n - 1;
    y = [];
    u = [];
    v = [];
    info = [];
    runhist = [];
    fprintf('\n Terminated at the initial point, primal KKT residual = %2.2e',pkkt);
    return;
end
%% main algorithm
parmain.tstart = tstart;
parmain.stoptol = stoptol;
parmain.stopop = stopop;
parmain.printyes = printyes;
parmain.maxiter = maxiter;
parmain.approxL = approxL;
parmain.approxRank = approxRank;
parmain.sigma = sigma;
parmain.m = m;
parmain.n = n;
[~,xnew,ynew,unew,vnew,info_main,runhist_main] = DualALM_main(LL,parmain,xnew,ynew,unew,vnew);
ttime = etime(clock,tstart);
iter = info_main.iter;
msg = info_main.msg;
if iter == maxiter
    msg = ' maximum iteration reached';
end
x = xnew;
y = ynew./s;
u = unew.*s;
v = vnew.*s;
%% compute original KKT residual
Lx = L*x;
Lxorg = Lx./s;
Rp = Lxorg - y;
normy = norm(y);
primfeas = max(norm(Rp)/normy,norm(min(x,0))/norm(x));
Rd = max((vnew'*L)' - n,0);
normu = norm(u);
dualfeas = max(norm(Rd)/(n),norm(u - v)/normu);
maxfeas = max(primfeas,dualfeas);
eta = norm(y - 1./v)/normy;
%% compute objective values
primobj = sum(x) + sum(log(s) - log(Lx))/n - 1;
dualobj = sum(log(v))/n;
obj = [primobj,dualobj];
gap = primobj - dualobj;
relgap = abs(gap)/(1 + abs(primobj) + abs(dualobj));
tmp = ((1./Lx')*L)'/n -1;
pkkt = norm(x - max(x + tmp,0));
pkkt2 = max(tmp);
%% Record infomation
runhist = runhist_main;
info.relgap = relgap;
info.iter = iter;
info.itersub = sum(runhist.itersub);
info.time = ttime;
info.timessn = sum(runhist.ttimessn);
info.eta = eta;
info.obj = obj;
info.maxfeas = maxfeas;
info.kktres = max(maxfeas,eta);
info.pkkt = pkkt;
info.pkkt2 = pkkt2;
info.sumlogLx = -sum(log(Lxorg));
info.count_L = info_main.count_L;
info.count_LT = info_main.count_LT;
if printyes
    fprintf('\n****************************************\n');
    fprintf([' ALM          : ',msg,'\n']);
    fprintf(' iteration    : %d\n',iter);
    fprintf(' L operator   : %d\n',info.count_L);
    fprintf(' LT operator  : %d\n',info.count_LT);
    fprintf(' time         : %3.2f\n',ttime);
    fprintf(' prim_obj     : %4.8e\n',primobj);
    fprintf(' dual_obj     : %4.8e\n',dualobj);
    fprintf(' relgap       : %4.5e\n',relgap);
    fprintf(' primfeas     : %3.2e\n',primfeas);
    fprintf(' dualfeas     : %3.2e\n',dualfeas);
    fprintf(' eta          : %3.2e\n',eta);
    fprintf(' primalKKT    : %3.2e\n',pkkt);
    fprintf(' primalKKT2   : %3.2e\n',pkkt2);
    fprintf(' -sum(log(Lx)): %1.8e\n',info.sumlogLx);
    fprintf(' sparsity     : %d\n',nnz(x));
end

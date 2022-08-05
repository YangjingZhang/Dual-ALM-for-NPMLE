function [obj,x,y,u,v,info,runhist] = DualALM_main(LL,parmain,x,y,u,v)
tstart = parmain.tstart;
stoptol = parmain.stoptol;
stopop = parmain.stopop;
printyes = parmain.printyes;
maxiter = parmain.maxiter;
approxL = parmain.approxL;
approxRank = parmain.approxRank;
sigma = parmain.sigma;
m = parmain.m;
n = parmain.n;
stop = 0;
sigmamax = 1e7;
sigmamin = 1e-8;
count_L = 0;
count_LT = 0;
%% initial objective values and feasibilities
Lx = LL.times(x);
count_L = count_L + 1;
obj(1) = sum(x) - sum(log(Lx))/n - 1;
obj(2) = sum(log(u))/n;
relgap = abs(obj(1) - obj(2))/(1 + abs(obj(1)) + abs(obj(2)));
Rp = Lx - y;
normy = norm(y);
primfeas = max(norm(Rp)/normy,norm(min(x,0))/norm(x));
LTv = LL.trans(v);
count_LT = count_LT + 1;
Rd = max(LTv - n,0);
normu = norm(u);
dualfeas = max(norm(Rd)/(n),norm(u - v)/normu);
maxfeas = max(primfeas,dualfeas);
eta = norm(y - 1./v)/normy;
if printyes
    fprintf('\n (dimension: m = %d, n = %d, ',m,n);
    fprintf('tol = %1.1e)\n',stoptol);
    fprintf('---------------------------------------------------');
    fprintf('\n iter|  [pinfeas   dinfeas  complem]    relgap|');
    fprintf('      pobj           dobj     |  time  sigma');
    fprintf('\n*********************************************');
    fprintf('*******************************************************');
    fprintf('\n %5.0f| [%3.2e %3.2e %3.2e] %- 3.2e| %- 8.6e  %- 8.6e |',...
            0,primfeas,dualfeas,eta,relgap,obj(1),obj(2));
    fprintf(' %5.1f| %3.2e|',etime(clock,tstart),sigma);
end
parNCG.tolconst = 0.5;
parNCG.count_L = count_L;
parNCG.count_LT = count_LT;
parNCG.approxL = approxL;
parNCG.approxRank = approxRank;
parNCG.m = m;
parNCG.n = n;
maxitersub = 20;
ssncgop.tol = stoptol;
ssncgop.printyes = printyes;
%% main loop
for iter = 1:maxiter
    parNCG.iter = iter;
    parNCG.sigma = sigma;    
    if dualfeas < 1e-5
        maxitersub = max(maxitersub,35);
    elseif dualfeas < 1e-3
        maxitersub = max(maxitersub,30);
    elseif dualfeas < 1e-1
        maxitersub = max(maxitersub,30);
    end
    ssncgop.maxitersub = maxitersub;
    %% SSN
    tstart_ssn = clock;
    [x,y,u,v,Lx,LTv,parNCG,~,info_NCG] = ...
        MLE_SSNCG(LL,x,y,v,LTv,parNCG,ssncgop);
    ttimessn = etime(clock,tstart_ssn);
    if (info_NCG.breakyes < 0)
        parNCG.tolconst = max(parNCG.tolconst/1.06,1e-3);
    end
    %% compute KKT residual
    Rp = Lx - y;
    normy = norm(y);
    primfeas = max(norm(Rp)/normy,norm(min(x,0))/norm(x));
    Rd = max(LTv - n,0);
    normu = norm(u);
    dualfeas = max(norm(Rd)/(n),norm(u - v)/normu);
    maxfeas = max(primfeas,dualfeas);
    eta = norm(y - 1./v)/normy;
    %% compute objective values
    primobj = sum(x) - sum(log(Lx))/n - 1;
    dualobj = sum(log(u))/n;
    obj = [primobj,dualobj];
    gap = primobj - dualobj;
    relgap = abs(gap)/(1 + abs(primobj) + abs(dualobj));   
    if (stopop == 1) && (maxfeas < stoptol) && (eta < stoptol)
        stop = 1;
    elseif (stopop == 2) && (eta < stoptol*10 || maxfeas < stoptol*10)
        pkkt = norm(x - max(x + LL.trans(1./(Lx))/n - 1,0));
        parNCG.count_LT = parNCG.count_LT + 1;
        if (pkkt < stoptol)
            stop = 1;
        end
    elseif (stopop == 3) && (eta < stoptol*10 || maxfeas < stoptol*10)
        tmp = LL.trans(1./(Lx))/n - 1;
        parNCG.count_LT = parNCG.count_LT + 1;
        pkkt = norm(x - max(x + tmp,0));        
        pkkt2 = max(tmp);
        if (max(pkkt2,pkkt) < stoptol)
            stop = 1;
        end
    elseif stopop == 4
        tmp = LL.trans(1./(Lx))/n - 1;
        parNCG.count_LT = parNCG.count_LT + 1;
        pkkt = norm(x - max(x + tmp,0)); 
        if (pkkt < stoptol)
            stop = 1;
        end
    end
    ttime = etime(clock,tstart);
    runhist.primfeas(iter)    = primfeas;
    runhist.dualfeas(iter)    = dualfeas;
    runhist.sigma(iter)       = sigma;
    runhist.primobj(iter)     = primobj;
    runhist.dualobj(iter)     = dualobj;
    runhist.gap(iter)         = gap;
    runhist.relgap(iter)      = relgap;
    runhist.ttime(iter)       = ttime;
    runhist.ttimessn(iter)    = ttimessn;
    runhist.itersub(iter)     = info_NCG.itersub - 1;
    runhist.iterCG(iter)      = info_NCG.tolCG;
    if printyes
        fprintf('\n %5.0d| [%3.2e %3.2e %3.2e] %- 3.2e| %- 8.6e  %- 8.6e |',...
            iter,primfeas,dualfeas,eta,relgap,primobj,dualobj);
        fprintf(' %5.1f| %3.2e|',ttime,sigma);
    end
    %% check termination
    if (stop && iter > 5) || (iter == maxiter)
        if stop
            termination = 'converged';
        elseif iter == maxiter
            termination = 'maxiter reached'; 
        end
        runhist.termination = termination;       
        runhist.iter = iter;    
        obj(1) = primobj;
        obj(2) = dualobj;
        break;
    end
    if (info_NCG.breakyes >=0) %%important to use >= 0
        sigma = max(sigmamin,sigma/10);
    elseif (iter > 1 && runhist.dualfeas(iter)/runhist.dualfeas(iter-1) > 0.6)
        if sigma < 1e7 && primfeas < 100*stoptol 
            sigmascale = 3;
        else
            sigmascale = sqrt(3);
        end
        sigma = min(sigmamax,sigma*sigmascale);
    end
end
info.maxfeas = maxfeas;
info.eta = eta;
info.iter = iter;
info.relgap = relgap;
info.ttime = ttime;
info.termination = termination;
info.msg = termination;
info.count_L = parNCG.count_L;
info.count_LT = parNCG.count_LT;
end
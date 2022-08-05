function [x,y,u,v,Lx,LTv,par,runhist,info] = MLE_SSNCG(LL,x,y,v,LTv,par,options)
printyes = options.printyes;
maxitersub = options.maxitersub;
tol = options.tol;
breakyes = 0;
maxitpsqmr = 500;
precond = 0;
stagnate_check_psqmr = 0;
sigma = par.sigma;
tiny = 1e-10;
n = par.n;
v1input = v - (n/sigma)*y;
[prox_v1,M_v1,~,prox_v1_prime_m] = prox_h(v1input,sigma/(n^2));
v2input = LTv/n + x/sigma - 1;
prox_v2 = max(v2input,0);
Lprox_v2 = LL.times(prox_v2);
par.count_L = par.count_L + 1;
phi = -(M_v1 + (sigma/2)*norm(prox_v2)^2);
par.precond = precond;
par.printyes = printyes;
%% main Newton iteration
for itersub = 1:maxitersub
    tmp = (sigma/n)*(v1input - prox_v1);
    Grad = (tmp + sigma*Lprox_v2)/n;
    normGrad = norm(Grad);    
    priminf_sub = normGrad/(norm(tmp)/n); 
    normu = norm(prox_v1);
    dualinf_sub = max(norm(max(LTv - n,0))/(n),norm(prox_v1 - v)/normu);
    if max(priminf_sub,dualinf_sub) < tol
        tolsubconst = 0.09;
    else
        tolsubconst = 0.005;
    end
    tolsub = max(min(1e-2,par.tolconst*dualinf_sub),tolsubconst*tol);
    runhist.priminf(itersub) = priminf_sub;
    runhist.dualinf(itersub) = dualinf_sub;
    runhist.phi(itersub) = phi;
    if printyes
        fprintf('\n      %2.0d  %- 11.10e  %3.2e %3.2e  %1.2e',...
            itersub,phi,priminf_sub,dualinf_sub,par.tolconst);
    end    
    if (priminf_sub < tolsub) && (itersub > 1)
        msg = 'good termination in subproblem:';
        if printyes
            fprintf('\n       %s  ',msg);
            fprintf(' dualinf = %3.2e, normGrad = %3.2e, tolsub = %3.2e',...
                dualinf_sub,priminf_sub,tolsub);
        end
        u = prox_v1;
        x = sigma*prox_v2;
        Lx = sigma*Lprox_v2;
        y = sigma/n*(prox_v1 - v1input);
        breakyes = -1;
        break;
    end
    %% compute Newton direction
    %% precond = 0,
    par.epsilon = min(1e-3,0.1*normGrad);    
    if (dualinf_sub > 1e-3) || (itersub <= 5)
        maxitpsqmr = max(maxitpsqmr,200);
    elseif (dualinf_sub > 1e-4)
        maxitpsqmr = max(maxitpsqmr,300);
    elseif (dualinf_sub > 1e-5)
        maxitpsqmr = max(maxitpsqmr,400);
    elseif (dualinf_sub > 5e-6)
        maxitpsqmr = max(maxitpsqmr,500);
    end
    if (dualinf_sub > 1e-4)
        stagnate_check_psqmr = max(stagnate_check_psqmr,20);
    else
        stagnate_check_psqmr = max(stagnate_check_psqmr,30);
    end
    if itersub > 3 && all(runhist.solve_ok(itersub - (3:-1:1)) <= -1) && dualinf_sub < 5e-5
        stagnate_check_psqmr = max(stagnate_check_psqmr,80);
    end
    par.stagnate_check_psqmr = stagnate_check_psqmr;
    if itersub > 1
        prim_ratio = priminf_sub/runhist.priminf(itersub - 1);
        dual_ratio = dualinf_sub/runhist.dualinf(itersub - 1);
    else
        prim_ratio = 0; dual_ratio = 0;
    end
    rhs = -Grad;
    if par.iter < 2 && itersub < 5
        tolpsqmr = min(1e-1,0.01*priminf_sub);
    else
        tolpsqmr = min(1e-1,0.001*priminf_sub);
    end
    const2 = 1;
    if itersub > 1 && (prim_ratio > 0.5 || priminf_sub > 0.1*runhist.priminf(1))
        const2 = 0.5*const2;
    end
    if dual_ratio > 1.1 
        const2 = 0.5*const2; 
    end
    tolpsqmr = const2*tolpsqmr;
    par.tol = tolpsqmr; 
    par.maxit = maxitpsqmr; 
    par.minitpsqmr = 5;
    %% find Newton direction
    [dv,resnrm,solve_ok,par] = Linsolver_MLE(rhs,LL,prox_v1_prime_m,v2input,par);
    iterpsqmr = length(resnrm) - 1;
    if printyes
        fprintf('| %3.1e %3.1e %3.0d %4d',par.tol,resnrm(end),iterpsqmr,par.r);
        fprintf(' %2.1f',const2);
    end   
    %% line search
    if (itersub <= 3) && (dualinf_sub > 1e-4) || (par.iter <= 3)
        stepop = 1;
    else
        stepop = 2;
    end
    steptol = 10*1e-5; 
    LTdv = LL.trans(dv);
    par.count_LT = par.count_LT + 1;
    [phi,v1input,prox_v1,prox_v1_prime_m,v2input,prox_v2,Lprox_v2,alp,iterstep,par] = ...
        findstep(Grad,dv,LTdv,LL,phi,v1input,prox_v1,prox_v1_prime_m,v2input,prox_v2,Lprox_v2,steptol,stepop,par);
    v = v + alp*dv;
    LTv = LTv + alp*LTdv;
    runhist.solve_ok(itersub) = solve_ok;
    runhist.psqmr(itersub)    = iterpsqmr;
    runhist.findstep(itersub) = iterstep;
    if alp < tiny
        breakyes = 11; 
    end
    phi_ratio = 1;
    if itersub > 1
        phi_ratio = (phi - runhist.phi(itersub - 1))/(abs(phi) + eps);
    end
    if printyes
        fprintf(' %3.2e %2.0f',alp,iterstep);
        if phi_ratio < 0 
            fprintf('-'); 
        end
    end
    %% check for stagnation
    printsub = printyes;
    if (itersub > 4)
        idx = max(1,itersub-3):itersub;
        tmp = runhist.priminf(idx);
        ratio = min(tmp)/max(tmp);
        if (all(runhist.solve_ok(idx) <= -1)) && (ratio > 0.9) ...
                && (min(runhist.psqmr(idx)) == max(runhist.psqmr(idx))) ...
                && (max(tmp) < 5*tol)
            fprintf('#')
            breakyes = 1;
        end
        const3 = 0.7;
        priminf_1half  = min(runhist.priminf(1:ceil(itersub*const3)));
        priminf_2half  = min(runhist.priminf(ceil(itersub*const3)+1:itersub));
        priminf_best   = min(runhist.priminf(1:itersub-1));
        priminf_ratio  = runhist.priminf(itersub)/runhist.priminf(itersub-1);
        dualinf_ratio  = runhist.dualinf(itersub)/runhist.dualinf(itersub-1);
        stagnate_idx   = find(runhist.solve_ok(1:itersub) <= -1);
        stagnate_count = length(stagnate_idx);
        idx2 = [max(1,itersub-7):itersub];
        if (itersub >= 10) && all(runhist.solve_ok(idx2) == -1) ...
                && (priminf_best < 1e-2) && (dualinf_sub < 1e-3)
            tmp = runhist.priminf(idx2);
            ratio = min(tmp)/max(tmp);
            if (ratio > 0.5)
                if (printsub); fprintf('##'); end
                breakyes = 2;
            end
        end
        if (itersub >= 15) && (priminf_1half < min(2e-3,priminf_2half)) ...
                && (dualinf_sub < 0.8*runhist.dualinf(1)) && (dualinf_sub < 1e-3) ...
                && (stagnate_count >= 3)
            if (printsub); fprintf('###'); end
            breakyes = 3;
        end
        if (itersub >= 15) && (priminf_ratio < 0.1) ...
                && (priminf_sub < 0.8*priminf_1half) ...
                && (dualinf_sub < min(1e-3,2*priminf_sub)) ...
                && ((priminf_sub < 2e-3) || (dualinf_sub < 1e-5 && priminf_sub < 5e-3)) ...
                && (stagnate_count >= 3)
            if (printsub); fprintf(' $$'); end
            breakyes = 4;
        end
        if (itersub >=10) && (dualinf_sub > 5*min(runhist.dualinf)) ...
                && (priminf_sub > 2*min(runhist.priminf)) %% add: 08-Apr-2008
            if (printsub); fprintf('$$$'); end
            breakyes = 5;
        end
        if (itersub >= 20)
            %% add: 12-May-2010
            dualinf_ratioall = runhist.dualinf(2:itersub)./runhist.dualinf(1:itersub-1);
            idx = find(dualinf_ratioall > 1);
            if (length(idx) >= 3)
                dualinf_increment = mean(dualinf_ratioall(idx));
                if (dualinf_increment > 1.25)
                    if (printsub); fprintf('^^'); end
                    breakyes = 6;
                end
            end
        end
        
    end    
end
if itersub == maxitersub
    u = prox_v1;
    x = sigma*prox_v2;
    Lx = sigma*Lprox_v2;
    y = sigma/n*(prox_v1 - v1input);
end
info.tolCG = sum(runhist.psqmr);
info.breakyes = breakyes;
info.itersub = itersub;
end
function [phi,v1input,prox_v1,prox_v1_prime_m,v2input,prox_v2,Lprox_v2,alp,iter,par] = ...
    findstep(Grad,dv,LTdv,LL,phi0,v1input0,prox_v10,prox_v1_prime_m0,v2input0,prox_v20,Lprox_v20,tol,options,par)
sigma = par.sigma;
n = par.n;
m = par.m;
printyes = par.printyes;
maxit = ceil(log(1/(tol + eps))/log(2));
c1 = 1e-4;
c2 = 0.9;
g0 = -(Grad'*dv);
if g0 <= 0
    if printyes
        fprintf('\n Need an ascent direction, %2.1e  ',g0);
    end    
    phi = phi0;
    v1input = v1input0;
    prox_v1 = prox_v10;
    prox_v1_prime_m = prox_v1_prime_m0;
    v2input = v2input0;
    prox_v2 = prox_v20;
    Lprox_v2 = Lprox_v20;
    alp = 0;
    iter = 0;
    return;
end
alp = 1;
alpconst = 0.5;
for iter = 1:maxit
    if iter == 1
        alp = 1; 
        LB = 0; 
        UB = 1;
    else
        alp = alpconst*(LB + UB);
    end    
    v1input = v1input0 + alp*dv;
    v2input = v2input0 + (alp/n)*LTdv;
    [prox_v1,M_v1,~,prox_v1_prime_m] = prox_h(v1input,sigma/(n^2));
    prox_v2 = max(v2input,0); 
    phi = -(M_v1 + (sigma/2)*norm(prox_v2)^2);
    tmp = (sigma/(n^2))*(v1input - prox_v1);
    galp = -tmp'*dv -(sigma/n)*(prox_v2'*LTdv);
    if iter == 1
        gLB = g0; gUB = galp;
        if sign(gLB)*sign(gUB) > 0
            if printyes
               fprintf('|');  
            end
            break;
        end
    end    
    if (abs(galp) < c2*abs(g0)) && (phi - phi0 - c1*alp*g0 > eps)%(phi - phi0 - c1*alp*g0 > 1e-15/max(1,abs(phi0)))
        if (options == 1) || ((options == 2) && (abs(galp) < tol))
            if printyes 
               fprintf(':'); 
            end
            break;
        end
    end   
    if sign(galp)*sign(gUB) < 0
        LB = alp; gLB = galp;
    elseif sign(galp)*sign(gLB) < 0
        UB = alp; gUB = galp;
    end
end
Lprox_v2 = LL.times(prox_v2);
par.count_L = par.count_L + 1;
if printyes && (iter==maxit)
    fprintf('m'); 
end
end     
    
    
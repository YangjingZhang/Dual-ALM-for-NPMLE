function [dv,resnrm,solve_ok,par] = Linsolver_MLE(rhs,LL,prox_v1_prime_m,v2,par)
n = par.n;
m = par.m;
sigma = par.sigma;
J = (v2 > 0);
r = sum(J);
par.r = r;
solveby = 'pcg';  % iterative solver cg
if n <= 5000
    solveby = 'pdirect';  % direct solver
end
if (r < 2000) || (n > 5000 && r < 5000)
    solveby = 'ddirect';  % woodbury formula  
end
if strcmp(solveby,'pdirect')
    if par.approxL
        U = LL.U;
        V = LL.V;
        VJ = V(J,:);
        LLT = U*(VJ'*VJ)*U';
        for i = 1:n
            LLT(i,i) = LLT(i,i) + prox_v1_prime_m(i);% + 1e-15;
        end
        cholLLT = mycholAAt(LLT,n);
    else
        LJ = LL.matrix(:,J);
        LLT = LJ*LJ';
        for i = 1:n           
            LLT(i,i) = LLT(i,i) + prox_v1_prime_m(i);% + 1e-15;
        end   
        cholLLT = mycholAAt(LLT,n);     
    end
    dv = mylinsysolve(cholLLT,rhs*(n^2/sigma));
    resnrm = 0;
    solve_ok = 1;
end
if strcmp(solveby,'pcg')
    if par.approxL
        U = LL.U;
        V = LL.V;
        VJ = V(J,:);
        Afun = @(v) ((prox_v1_prime_m.*v + U*((VJ*(v'*U)')'*VJ)')*(sigma/n^2));
    else
        LJ = LL.matrix(:,J);
        Afun = @(v) ((prox_v1_prime_m.*v + LJ*(LJ'*v))*(sigma/n^2));
    end
    [dv,~,resnrm,solve_ok] = psqmr(Afun,rhs,par);
end
if strcmp(solveby,'ddirect')
    rhs = rhs*(n^2/sigma);
    prox_v1_prime_m = prox_v1_prime_m + eps;
    if par.approxL
        U = LL.U;
        V = LL.V;
        VJ = V(J,:);
        rhstmp = VJ*((rhs./prox_v1_prime_m)'*U)';
        LTL = VJ*(U'*((U./prox_v1_prime_m)))*VJ';
        for i = 1:r
            LTL(i,i) = LTL(i,i) + 1;
        end
        if r <= 1000
            dv = LTL\rhstmp;
        else
            cholLTL = mycholAAt(LTL,r);
            dv = mylinsysolve(cholLTL,rhstmp);
        end
        dv = (U*(dv'*VJ)')./prox_v1_prime_m;
        dv = rhs./prox_v1_prime_m - dv; 
    else
        if r == m
            LJ = LL.matrix;
        else
            LJ = LL.matrix(:,J);
        end    
        LJ2 = LJ./prox_v1_prime_m;
        rhstmp = (rhs'*LJ2)';
        LTL = LJ'*LJ2;
        LTL = eye(r) + LTL;
        if r <= 1000
            dv = LTL\rhstmp;
        else
            cholLTL = mycholAAt(LTL,r);
            dv = mylinsysolve(cholLTL,rhstmp);
        end
        dv = LJ2*dv;
        dv = rhs./prox_v1_prime_m - dv;    
    end
    resnrm = 0;
    solve_ok = 1;
end
end
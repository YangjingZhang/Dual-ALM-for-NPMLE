%% GMLEB estimator
%% Input the matrix L \in R^{n*m}
%%       vector x in R^m, x >= 0, sum xi = 1
%%       grid points U \in R^{m*d}
%% Output estimator theta_hat \in R^{n*d}
%% Yangjing Zhang 2021
function theta_hat = EB_estimator(L,x,U)
Lx = L*x;
[n,~] = size(L);
d = size(U,2);
theta_hat = zeros(n,d);
for i = 1:n
    theta_hat(i,:) = (x'*(L(i,:)'.*U))/Lx(i);
end

end
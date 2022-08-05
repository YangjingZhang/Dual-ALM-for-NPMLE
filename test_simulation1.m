%% Simulation 1 (d=1)
%% The experiment is a replication of that in Johnston and Silverman (2004),
%% Brown and Greenshtein (2009),  Jiang and Zhang (2009).
%% Settings: there are n observations.
%% theta_i = mu, i = 1,...,k, theta_i = 0, i = k+1,...,n
%% observations are generated following Xi ~ N(theta_i,1).
%% m grid points are taken as in Sec 2.3 of Jiang and Zhang (2009). Basically,
%% m equally spaced points are choosen in the interval [min Xi, max Xi]
%%
%% n #observations
%% k #nonzero theta_i
%% mu nonzero theta_i = mu
%% m #grid points
addpath(genpath(pwd));
rng(1);
clear;
n = 1000;
m = 500;
k = 500;
mu = 7;
%% observations
obser = zeros(n,1);
obser(1:k) = randn(k,1) + mu;
obser(k + 1:n) = randn(n - k,1);
%% grid points
ub = max(obser) + eps;
lb = min(obser) - eps;
grid = linspace(lb,ub,m);
grid = grid';
%% L
diffM = obser*ones(1,m) - ones(n,1)*grid';
L = normpdf(diffM);
%% solver
options.maxiter = 100;
options.stoptol = 1e-6;
options.stopop = 3;
options.printyes = 1;
options.approxL = 1;
[obj,x,y,u,v,info,runhist] = DualALM(L,options);
%% plot figure
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
lw = 2;
ms = 4;
fs = 15;
% f true and f_hat
figure(1);
versionold = verLessThan('matlab','9.7');
if versionold
    subplot(131);
else
    t = tiledlayout(1,3,'Padding','none','TileSpacing','none');
    nexttile;
end
xmax = max(grid) + 1;
xx = (- xmax - 1):0.01:(xmax + 1);
yy = (k/n)*normpdf(xx,mu,1) + (n - k)/n*normpdf(xx,0,1);
plot(xx,yy,'linewidth',lw,'markersize',ms);
hold on;
[obsersort,id] = sort(obser);
plot(obsersort,y(id),':','linewidth',lw,'markersize',ms);
legend('True density $f_{G^*,1}$','Estimated mixture density $\widehat{f}_{\widehat{G}_n,1}$','location','northwest','fontsize',fs);
yl = get(gca,'ylim');
ylim([min(yl(1),-0.01),yl(2)+0.06]);
axis square; box on; hold off;
% mixing measure
if versionold
    subplot(132);
else
    nexttile;
end
plot(grid,x,'linewidth',lw,'markersize',ms);
legend('NPMLE of the prior probability measure $\widehat{G}_n$','location','northwest','fontsize',fs);
yl = get(gca,'ylim');
ylim([min(yl(1),-0.02),yl(2) + 0.1]);
axis square; box on;
% Tweedie formula/Bayes rule
if versionold
    subplot(133);
else
    nexttile;
end
theta_hat = L*(grid.*x)./(L*x);
plot(obsersort,theta_hat(id),'-+','linewidth',lw,'markersize',ms);
legend('Bayes estimator of $\theta_i$','location','northwest','fontsize',fs);
yl = get(gca,'ylim');
ylim([min(yl(1),-0.01),yl(2)]);
axis square; box on;
set(gcf,'Position',[50 50 1800 600]);

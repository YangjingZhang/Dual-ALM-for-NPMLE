%% simulation 2
%% d=2
%% fig_option = 1:
%%          theta_i drawn from two circles,
%%          from circles of radius 2, i = 1,...,n/2
%%          from circles of radius 6, i = n/2+1,...,n
%% fig_option = 2:
%%          theta_i drawn from a triangle
%% fig_option = 3:
%%          theta_i drawn from a digit 8
%% fig_option = 4:
%%          theta_i drawn from letter A
%% sigma_option = 1, SIGMA=I2,
%%              = 2, SIGMA=diag(a,b),a,b ~ Uniform[1,3]
%% observations are generated following Xi ~ N(theta_i,SIGMA).
%% m grid points:
%% grid_option = 1:
%%          m = n, grid points are chosen to be the data points X1,...,Xn
%% grid_option = 2:
%%          m grid points are taken as a random subsample of the data points X1,...,Xn
%% grid_option = 3:
%%          use a uniform grid in a compact region (e.g.,rectangle in d=2) containing the data
clear;
close all;
addpath(genpath(pwd));
n = 5000;
m = 5000;
d = 2;
methodtype = 'ALM'; % 'EM' 'PEM'
fig_option = 1;
sigma_option = 1;
grid_option = 1;
%% observations
[obser,theta,SIGMA] = generate_observation(n,fig_option,sigma_option,d);
%% grid points
[grid0,mnew] = select_grid(obser,grid_option,m);
%% L
[L,rowmax,removeind] = likelihood_matrix(obser,grid0,SIGMA,1);
if ~isempty(removeind)
    n = size(L,1);
end
%% solver
if strcmp(methodtype,'ALM')
    options.scaleL = 0;
    options.approxL = 0;
    options.stoptol = 1e-6;
    options.printyes = 1;
    tic;
    [~,x,~,~,~,info,~] = DualALM(L,options);
    runt = toc;
    L = likelihood_matrix(obser,grid0,SIGMA,0);
    llk = sum(log(L*x))/n;
    iter = info.iter;
    fprintf('iter = %d, sum(log(Lx))/n = %5.8e \n',iter,llk);
elseif strcmp(methodtype,'EM')
    options.printyes = 1;
    options.stoptol = 1e-4;
    tic;
    [x,grid0,~,iter] = EM(obser,SIGMA,m,options);
    runt = toc;
    L = likelihood_matrix(obser,grid0,eye(d),0);
    llk = sum(log(L*x))/n;
    fprintf('iter = %d, sum(log(Lx))/n = %5.8e \n',iter,llk);
elseif strcmp(methodtype,'PEM')
    options.stoptol = 1e-4;
    options.printyes = 0;
    tic;
    [x,grid0,~,iter] = PEM(obser,SIGMA,m,options);
    runt = toc;
    L = likelihood_matrix(obser,grid0,eye(d),0);
    llk = sum(log(L*x))/n;
    fprintf('iter = %d, sum(log(Lx))/n = %5.8e \n',iter,llk);
    llk0 = llk;
    L0 = L;
end
%% plot results
% GMLEB estimator
theta_hat = EB_estimator(L,x,grid0);
mse = norm(theta - theta_hat,'fro')^2/n;
plot_yes = [1 1 1];%raw,EB,Grid
if sum(plot_yes) > 0
    set(groot,'defaultAxesTickLabelInterpreter','latex');
    set(groot,'defaulttextinterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');
    x = x/sum(x);
    ms = 4;
    fs = 20;
    tiny = 0;
    xmax = 9.5; xmin = -9.5;
    ymax = 9.5; ymin = -9.5;
    x_tic = -8:2:8;
    y_tic = -8:2:8;
switch fig_option
    case 1
        xmin = -9.5; xmax = 9.5;
        ymin = -9.5; ymax = 9.5;
        x_tic = -8:2:8;
        y_tic = -8:2:8;
    case 2
        xmin = -6; xmax = 6;
        ymin = -3; ymax = 9;
        x_tic = -4:2:4;
        y_tic = -2:2:8;
    case 3
        xmin = -9.5; xmax = 9.5;
        ymin = 3 - 9.5; ymax = 3 + 9.5;
        x_tic = -8:2:8;
        y_tic = -6:2:12;
    case 4
        xmin = -9.5; xmax = 9.5;
        ymin = -9.5; ymax = 9.5;
        x_tic = -8:2:8;
        y_tic = -8:2:8;
end
    yL = [ymin,ymax];
    xL = [xmin,xmax];
    x_txt = xL(1) + 0.03*(xL(2) - xL(1));
    y_txt = yL(2) - 0.06*(yL(2) - yL(1));
    y_txt1 = yL(1) + 0.2*(yL(2) - yL(1));
    y_txt2 = yL(1) + 0.13*(yL(2) - yL(1));
    y_txt3 = yL(1) + 0.06*(yL(2) - yL(1));
    figure(1);
    versionold = verLessThan('matlab','9.7');
    ppp = 1; kkk = 0;
    if plot_yes(1)
        if versionold
            subplot(1,sum(plot_yes),ppp); ppp = ppp + 1;
        else
            tiledlayout(1,sum(plot_yes),'Padding','none','TileSpacing','none'); kkk = 1;
            nexttile;
        end
        % 1
        plot(theta(:,1),theta(:,2),'k.','markersize',ms);%True Signal
        hold on;
        plot(obser(:,1),obser(:,2),'.','color','b','markersize',ms);%Raw Data
        % dummy graph for legend only
        k11 =  plot(1e3,1e3,'k.','markersize',fs);
        k22 =  plot(1e3,1e3,'.','color','b','markersize',fs);
        legend([k11 k22],{'True Signal','Raw Data'},'FontSize',fs,'Location','northwest');
        xlim(xL);
        ylim(yL);
        xticks(x_tic); yticks(y_tic);
        legend boxoff; axis square; box on; hold off;
    end
    if plot_yes(2)
        %  GMLEB + True Signal
        if versionold
            subplot(1,sum(plot_yes),ppp); ppp = ppp + 1;
        else
            if kkk == 0
                tiledlayout(1,sum(plot_yes),'Padding','none','TileSpacing','none');
            end
            nexttile;
        end
        plot(theta(:,1),theta(:,2),'k.','markersize',ms);
        hold on;
        plot(theta_hat(:,1),theta_hat(:,2),'.','color','r','markersize',ms);
        xlim(xL);
        ylim(yL);
        xticks(x_tic);
        if kkk == 0
            yticks(y_tic);
        else
            set(gca,'yticklabel',[]);
        end
        axis square; box on; hold off;
        text(x_txt,y_txt,'Empirical Bayes','fontsize',fs);
    end
    if plot_yes(3)
        % 4 G_n_hat
        if versionold
            subplot(1,sum(plot_yes),ppp); ppp = ppp + 1;
        else
            nexttile
        end
        scl = 50;
        tiny = 0;%1e-4;
        mx = max(x);
        if strcmp(methodtype,'EM')
            tiny = 1e-4; mx = 1/20;
        end
        for i = 1:m
            if x(i) > tiny
                plot(grid0(i,1),(grid0(i,2)),'k.','markersize',x(i)*(scl/mx));
                hold on;
            end
        end
        plot(theta(:,1),theta(:,2),'k.','markersize',ms);

        xlim(xL);
        ylim(yL);
        xticks(x_tic);
        set(gca,'yticklabel',[]);
        axis square; box on; hold off;
        text(x_txt,y_txt,'$\widehat{G}_n$','fontsize',fs);
    end
    set(gcf,'Position',[50 50 500*sum(plot_yes) 500]);
end

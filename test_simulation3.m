%% simulation 3
%% d=3,4,5...
%% fig_option = 5,6,7,8,9
%% =5: the first 2 dimensions of theta_i are drawn from a circle
%%     circles of radius 6, i = 1,...,n
%%     the remaining dimensions of theta_i = 0
%% =6: theta_i=0
%% =7: theta_i has 3 support points, G^* = sum_{j=1}^3 (1/3) \delta_{6*x_j}
%%     x_j = [0 0 0 ... 0]
%%         = [1 0 0 ... 0]
%%         = [0 1 0 ... 0]
%% =8: theta_i ~ N(0,eye(d))
%% =9: theta_i has 6 support points
%%     [0  0 0 ... 0]
%%     [1  0 0 ... 0]
%%     [-1 0 0 ... 0]
%%     [0  1 0 ... 0]
%%     [1  1 0 ... 0]
%%     [-1 1 0 ... 0]
%% observations are generated following Xi ~ N(theta_i,eye(d)).
%% m = n, grid points are chosen to be the data points X1,...,Xn
clear;
close all;
addpath(genpath(pwd));
n = 5000;
m = n;
d = 9;
methodtype = 'PEM'; % 'ALM' 'EM' 'PEM'
fig_option = 7;%5 6 7 8 9
%% observations
[obser,theta,SIGMA] = generate_observation(n,fig_option,1,d);
if m < n
    grid0 = obser(randperm(n,m),:);
else
    grid0 = obser;
end
%% L
[L,rowmax,removeind] = likelihood_matrix(obser,grid0,SIGMA,1);
modifyL = 1;
if modifyL && (m == n)
    for kk = 1:n
        L(kk,kk) = 0;
        krow = L(kk,:);
        tmp = max(krow);
        L(kk,:) = krow/tmp;
    end
end
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
    L = likelihood_matrix(obser,grid0,eye(d),0);
    llk = sum(log(L*x))/n;
    iter = info.iter;
    fprintf('iter = %d, sum(log(Lx))/n = %5.8e \n',iter,llk);
elseif strcmp(methodtype,'EM')
    options.stoptol = 1e-4;
    SIGMA = ones([1 d n]);
    tic;
    [x,grid0,~,iter] = EM(obser,SIGMA,m,options);
    runt = toc;
    L = likelihood_matrix(obser,grid0,eye(d),0);
    llk = sum(log(L*x))/n;
    fprintf('iter = %d, sum(log(Lx))/n = %5.8e \n',iter,llk);
elseif strcmp(methodtype,'PEM')
    options.stoptol = 1e-4;
    options.printyes = 0;
    SIGMA = ones([1 d n]);
    tic;
    [x,grid0,~,iter] = PEM(obser,SIGMA,m,options);
    runt = toc;
    L = likelihood_matrix(obser,grid0,eye(d),0);
    llk = sum(log(L*x))/n;
    fprintf('iter = %d, sum(log(Lx))/n = %5.8e \n',iter,llk);
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
        case 7
            xmin = -3.5; ymin = xmin;
            x_tic = -2:2:8;
            y_tic = x_tic;
        case 9
            ymin = -3.5;
            y_tic = -2:2:8;
    end
    yL = [ymin,ymax];
    xL = [xmin,xmax];
    x_txt = xL(1) + 0.03*(xL(2) - xL(1));
    y_txt = yL(2) - 0.06*(yL(2) - yL(1));
    % project to (1-2) plane
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
        if fig_option >= 6
            grid on;
        end
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
        if fig_option >= 6
            grid on;
        end
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
        if fig_option >= 6
            grid on;
        end
        text(x_txt,y_txt,'$\widehat{G}_n$','fontsize',fs);
    end
    set(gcf,'Position',[50 50 500*sum(plot_yes) 500]);
end


%% select grid points
%% Input: observations X \in R^{n*d} (n observations in each row)
%%        grid_option = 1 grid points are chosen to be the data points
%%                = 2 grid points are taken as a random subsample of the data points
%%                = 3 use a uniform grid in a compact region (e.g.,rectangle in d=2) containing the data
%%                = 4 logspace of y, linspace of x
%% Output: grid points U \in R^{m*d} (m observations in each row)
%% Yangjing Zhang 2021
function [U,m] = select_grid(X,grid_option,m)
n = size(X,1);
if ~exist('grid_option','var') || isempty(grid_option)
    grid_option = 1;
end
switch grid_option
    case 1
        U = X;
        m = n;
    case 2        
        m = min(m,n);
        if ~exist('m','var')
            m = round(sqrt(n));
        end
        U = X(randperm(n,m),:);
    case 3
        if ~exist('m','var')
            m = n;
        end
        xmax = max(X(:,1));
        xmin = min(X(:,1));
        ymax = max(X(:,2));
        ymin = min(X(:,2));
        ratio = 1;%(xmax - xmin)/(ymax - ymin);
        my = round(sqrt(m/ratio));
        mx = round(sqrt(m*ratio));
        m = mx*my;
        xgrid = linspace(xmin,xmax,mx);
        ygrid = linspace(ymin,ymax,my);
        [Xg,Yg] = meshgrid(xgrid,ygrid);
        U = [Xg(:),Yg(:)];
    case 4
        if ~exist('m','var')
            m = n;
        end
        xmax = max(X(:,1));
        xmin = min(X(:,1));
        ymax = max(X(:,2));
        ymin = min(X(:,2));
        ratio = 1;%(xmax - xmin)/(ymax - ymin);
        my = round(sqrt(m/ratio));
        mx = round(sqrt(m*ratio));    
        m = mx*my;
        xgrid = linspace(xmin,xmax,mx);
        ygrid = logspace(log10(ymin),log10(ymax),my);
        [Xg,Yg] = meshgrid(xgrid,ygrid);
        U = [Xg(:),Yg(:)];
    otherwise
        U = X;
        m = n;
end
end


%% generate observations in R^2
%% Input: n # observations
%%        fig_option = 1 two concentric circles
%%                   = 2 triangle
%%                   = 3 digit 8
%%                   = 4 letter A
%%                   = 5 circle of radius 6
%%                   = 6 theta_i = 0
%%                   = 7 theta_i = 0, 6*e1, or 6*e2
%%                   = 8 theta_i = N(0,SIGMA);
%% sigma_option = 1, I2,
%%              = 2, SIGMA_i = diag(ai,bi),ai,bi ~ Uniform[1,3]
%% Output: observations X \in R^{n*d} (n observations in each row)
%%         theta \in R^{n*d} true signal
%%         SIGMA \in R^{1*d*n} if sigma_option = 2, SIGMA(:,:,i) = [ai,bi]
%% Yangjing Zhang 2021
function [X,theta,SIGMA] = generate_observation(n,fig_option,sigma_option,d)
if ~exist('d','var') || d < 2
    d = 2;
end
if ~exist('sigma_option','var') || sigma_option == 1
    SIGMA = eye(d);
elseif sigma_option == 2
    SIGMA = rand([1 d n])*2 + 1; %uniform in (1,3)
end
switch fig_option
    case 1
        r1 = 2;
        r2 = 6;
        n1 = round(n/2);
        n2 = n - n1;
        theta = zeros(n,d);
        t = (2*pi)*rand(n1,1);
        theta(1:n1,:) = r1*[cos(t),sin(t)];
        t = (2*pi)*rand(n2,1);
        theta(n1 + 1:n,:) = r2*[cos(t),sin(t)];
        X = mvnrnd(theta,SIGMA);
    case 2
        n1 = round(n/3);
        n2 = n1;
        n3 = n - n1 - n2;
        p1 = [-3 0];
        p2 = [0 6];
        p3 = [3 0];
        theta = [p1 + (p2 - p1).*rand(n1,1); p2 + (p3 - p2).*rand(n2,1); p3 + (p1 - p3).*rand(n3,1)];
        X = mvnrnd(theta,SIGMA);
    case 3
        r1 = 3; c1 = [0 0];
        r2 = 3; c2 = [0 6];
        n1 = round(n/2);
        n2 = n - n1;
        theta = zeros(n,d);
        t = (2*pi)*rand(n1,1);
        theta(1:n1,:) = c1 + r1*[cos(t),sin(t)];
        t = (2*pi)*rand(n2,1);
        theta(n1 + 1:n,:) = c2 + r2*[cos(t),sin(t)];
        X = mvnrnd(theta,SIGMA);
    case 4
        n1 = round(n/5); n2 = n1; n3 = n1; n4 = n1;
        n5 = n - (n1 + n2 + n3 + n4);
        p1 = [-4 -6];
        p2 = [-2 0];
        p3 = [0 6];
        p4 = [2 0];
        p5 = [4 -6];
        theta = [p1 + (p2 - p1).*rand(n1,1); p2 + (p3 - p2).*rand(n2,1); ...
            p3 + (p4 - p3).*rand(n3,1); p4 + (p5 - p4).*rand(n4,1); p4 + (p2 - p4).*rand(n5,1);];
        X = mvnrnd(theta,SIGMA);
    case 5
        r = 6;
        t = (2*pi)*rand(n,1);
        theta = zeros(n,d);
        theta(:,1:2) = r*[cos(t),sin(t)];
        X = mvnrnd(theta,SIGMA);
    case 6
        theta = zeros(n,d);
        X = mvnrnd(theta,SIGMA);
    case 7
        r = 6;
        atoms = zeros(3,d);
        atoms(2,1) = r; atoms(3,2) = r;
        theta = zeros(n,d);
        xstar = ones(3,1)/3;% equal weights
        xstar2 = cumsum(xstar);
        kk = 0;
        for i = 1:3
            kend = round(n*xstar2(i));
            theta((kk + 1):kend,:) = repmat(atoms(i,:),kend - kk,1);
            kk = kend;
        end
        X = mvnrnd(theta,SIGMA);
    case 8
        theta = mvnrnd(zeros(n,d),SIGMA);
        X = mvnrnd(theta,SIGMA);
    case 9
        r = 6;
        atoms = zeros(6,d);
        atoms(2,1) = r; 
        atoms(3,1) = -r;
        atoms(4,2) = r;
        atoms(5,1:2) = [r r];
        atoms(6,1:2) = [-r r];
        theta = zeros(n,d);
        xstar = ones(6,1)/6;% equal weights
        xstar2 = cumsum(xstar);
        kk = 0;
        for i = 1:6
            kend = round(n*xstar2(i));
            theta((kk + 1):kend,:) = repmat(atoms(i,:),kend - kk,1);
            kk = kend;
        end
        X = mvnrnd(theta,SIGMA);
end
end

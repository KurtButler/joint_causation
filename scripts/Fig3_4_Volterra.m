%% Control for the random seed
rng(0)

%% Training data
N = 500;
T = 8;

a = randn(1,T+1);
B = zeros(T+1);
B(1,2) = 1;
B(2,3) = 1;
B(4,3) = 1;
B(4,1) = -1;
B(5,4) = 1;
B(7,6) = -1;
B(8,5) = 1;
B = 0.5*(B+B');


t = (1:N)';
x = zeros(N,1);
y = zeros(N,1);
for n = 2:N
    x(n) = 0.85*x(n-1) + randn;

    if n>T
        m = x(n:-1:n-T);
        y(n) = a*m + m'*B*m;
    end
end

y = awgn(y,20,'measured');


figure(11)
tiledlayout(2,1,"TileSpacing","tight", 'Padding','tight')

nexttile
plot(t,x,'k','LineWidth',1)
grid minor;
% title('Input signal x_t','FontSize',15)
legend('x_t','FontSize',12)

nexttile
plot(t,y,'r','LineWidth',1)
grid minor;
% title('Output signal y_t','FontSize',15)
legend('y_t','FontSize',12)



set(gcf,'Position',[60 870 602 197])
saveas(gcf,'./results/two_signals.png')


%% Fitting models
xt = embed(x,T+1,1);
yt = lastcol(embed(y,T+1,1));

gp = fitrgp(xt,yt,'KernelFunction','ardsquaredexponential');
yp = zeros(size(xt,1),1);
yp(:) = predict(gp,xt);
dyp = 0*yp;
ddyp = 0*yp;

% Volterra model
H = xt;
P = size(xt,2);
betalookup = zeros(54,2);
for p = 1:P
    ctr = size(H,2);
    H = cat(2,H,xt(:,1:p).*xt(:,p));
    betalookup(ctr + (1:p),1) = 1:p;
    betalookup(ctr + (1:p),2) = p;
end


beta = pinv(H)*yt;
VoltHess = zeros(P);
for l = P+1:numel(beta)
    p = betalookup(l,1);
    q = betalookup(l,2);
    if p~=q
        VoltHess(p,q) = beta(l);
        VoltHess(q,p) = beta(l);
    else
        VoltHess(p,q) = 2*beta(l);
    end
end
VoltHess = rot90(VoltHess,2);



%% MDCE Analysis
HESS = zeros(numel(yp),(T+1)^2);
Best = zeros(T+1);
Hess = zeros(T+1);

for i1=1:T+1
    for i2=1:T+1
        alpha = gp.Alpha;
        l = gp.KernelInformation.KernelParameters(1:end-1);
        sf= gp.KernelInformation.KernelParameters(end);
        sg= gp.Sigma;

        [i1,i2]


        dx1 = (xt(:,i1)-xt(:,i1)')/(-l(i1)^2);
        dx2 = (xt(:,i2)-xt(:,i2)')/(-l(i2)^2);
        del12 = 1/(-l(i1)*l(i2));

        dkdx = (sf.^2)*exp(-0.5*(pdist2(xt./l',xt./l')).^2).*dx1;
        ddkdxdx = (sf.^2)*exp(-0.5*(pdist2(xt./l',xt./l')).^2).*(dx1.*dx2 + del12);

        dyp = dkdx*alpha;
        ddyp(:) = ddkdxdx*alpha;

        HESS(:,i1 + 2*(i2-1)) = ddyp(:);

        Best(T+2-i1,T+2-i2) = mean((ddyp));
        Hess(i1,i2) = B(i1,i2) + B(i2,i1);
    end
end

%% Bayesian hypothesis test 
omit = @(x,p) x(:,[1:p-1,p+1:size(x,2)]);

fprintf('Running the Bayes detector...\n')
BayesTest = zeros(9);
for p = 1:P
    fprintf('\n');
    for q = 1:p-1
        fprintf('.');
        kfcn = @(Xn,Xm,theta) kSE(omit(Xn,p),omit(Xm,p),theta(1:P)) +kSE(omit(Xn,q),omit(Xm,q),theta(P+1:2*P));
        theta0 = zeros(1,2*P);
        gptemp = fitrgp(xt,yt,'KernelFunction',kfcn,'KernelParameters',theta0);

        if gp.LogLikelihood >= gptemp.LogLikelihood
            BayesTest(p,q) = 1;
            BayesTest(q,p) = 1;
        end
    end
end


BayesTest = rot90(BayesTest,2);

%% Plotting
colnames = {};
for i = 1:T+1
    if i > 1
    colnames{i} = sprintf('x_{t-%d}',i-1);
    else
    colnames{i} = 'x_t';
    end
end

figure(12)
tiledlayout(2,2,"TileSpacing","compact", 'Padding','tight')

nexttile
imagesc(Hess)
title('','True Hessian Matrix','FontSize',15)
colorbar;
clim([-1,1])
o = (0:2:255)'; o =[0*o;o]; colors = (1/256)*[o,0*o, flipud(o)];
colormap(colors)
xticks(1:9)
xticklabels(colnames)
yticks(1:9)
yticklabels(colnames)


nexttile
imagesc(Best)
title('','Average MDCE','FontSize',15)
colorbar;
clim([-1,1])
xticks(1:9)
xticklabels(colnames)
yticks(1:9)
yticklabels(colnames)


nexttile
imagesc(VoltHess)
title('','Volterra model','FontSize',15)
colorbar;
clim([-1,1])
xticks(1:9)
xticklabels(colnames)
yticks(1:9)
yticklabels(colnames)



nexttile
imagesc(BayesTest)
title('','Bayes detector','FontSize',15)
colorbar;
clim([-1,1])
xticks(1:9)
xticklabels(colnames)
yticks(1:9)
yticklabels(colnames)


uu = (1/256)*linspace(-250,256,500)';
redblue = max([uu,flipud([uu,uu])],0);
colormap(redblue)


%% Save output
set(gcf,'Position',[95 285 514 441]);
saveas(gcf,'./results/hessian.png')


%% Functions
function [Mx] = embed(x,Q,tau)
% EMBED(x,Q,tau)
% Computes the Q-dimensional delay embedding of the signal x(t)
% with embedding delay tau.
% If x is an Nx1 column vector, then the result will be an
% (N - (Q-1)*tau)xQ matrix whose rows are points in the shadow manifold 
L = size(x,1) - (Q-1)*tau;
if size(x,2)>1
    if all(imag(x(:))==0)
        Mx = zeros(L, size(x,2)*Q);
        for k=1:size(x,2)
            Mx(:,(k-1)*Q +(1:Q)) = embed(x(:,k),Q,tau);
        end
    else
        Mx = [];
        for k=1:size(x,2)
            M = embed(x(:,k),Q,tau);
            Mx = [Mx, M];
        end
    end
else
    if all(imag(x(:))==0)
        Mx = zeros(L,0);
        for q = 1:Q
            Mx = cat(2,  Mx,  x((1:L)+(q-1)*tau,:)  );
        end
    else
        Mx = embed([real(x),imag(x)],Q,tau);
    end
end
end

function y = lastcol(Y)
    y = Y(:,end);
end

% Product of two periodic kernels
function K = kSE(Xm,Xn,theta)
    var12 = exp(theta(1));
    ll = exp(theta(1 + (1:size(Xm,2))))';
    distmat = pdist2(Xm./ll,Xn./ll);
    K = var12*exp(-0.5*distmat.^2);
end

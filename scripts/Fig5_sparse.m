%% Control for the random seed
rng(0)

%% Simulation
N = 1000;
x = 4*rand(N,2)-2;
f = @(x) sin(x*[2;2]).*x(:,2) + 3*cos(x(:,1)) + sin(x(:,2));
y = f(x);
y = awgn(y,20,'measured');


mdce_operator = @(f,x) 1e8*(f(x-[1e-4 1e-4]) - f(x-[0 1e-4])-f(x-[1e-4 0]) + f(x));


SUBSAMPLE = 1000;
gp = fitrgp(x(1:SUBSAMPLE,:),y(1:SUBSAMPLE,:));


D = size(x,2);
M = 200;
v = 0.1*randn(M,D);
H = [sin(x*v'), cos(x*v')]; 

% theta = pinv(H)*y;
mdl = fitlm(H,y,'Intercept',false,'RobustOpts','off');
theta = mdl.Coefficients.Estimate;

sparsefunc = @(x) [sin(x*v'), cos(x*v')]*theta;

sparsemdce = @(x) (-[sin(x*v'), cos(x*v')].*[v(:,1).*v(:,2); v(:,1).*v(:,2)]')*theta;


%% Plot results
figure(31)
tiledlayout(3,2,'Padding','tight','TileSpacing','tight')

[px1,px2] = meshgrid(linspace(-2,2,50));
py = zeros(size(px1));
py(:) = f([px1(:),px2(:)]);
py0 = py;

% True Function
nexttile
surf(px1,px2,py,'EdgeColor','none')
title('True Function','FontSize',15)
view([0 0 1])
colorbar
xlabel('x_1')
ylabel('x_2')



% True MDCE
py(:) = mdce_operator(f,[px1(:),px2(:)]);
py0 = py;
nexttile
surf(px1,px2,py,'EdgeColor','none')
title('True MDCE','FontSize',15)
clim([-10,10]);
view([0 0 1])
colorbar
xlabel('x_1')
ylabel('x_2')




% GPR Function
py(:) = predict(gp,[px1(:),px2(:)]);
MSE = mean((py(:)-py0(:)).^2);
nexttile
surf(px1,px2,py,'EdgeColor','none')
title('Function estimated by GPR',sprintf('N=%d, MSE=%0.3f',SUBSAMPLE,MSE),'FontSize',15)
view([0 0 1])
colorbar
xlabel('x_1')
ylabel('x_2')



% GPR MDCE
py(:) = getderivative(gp,[px1(:),px2(:)],x(1:SUBSAMPLE,:));
MSE = mean((py0(:) - py(:)).^2);
nexttile
surf(px1,px2,py,'EdgeColor','none')
title('MDCE estimated by GPR',sprintf('N=%d, MSE=%0.3f',SUBSAMPLE,MSE),'FontSize',15)
zlim([-10,10])
clim([-10,10]);
view([0 0 1])
colorbar
xlabel('x_1')
ylabel('x_2')






% Sparse Function
py(:) = sparsefunc([px1(:),px2(:)]);
MSE = mean((py(:)-py0(:)).^2);
nexttile
surf(px1,px2,py,'EdgeColor','none')
title('Function estimated by Sparse GPR',sprintf('N=%d, M=%d, MSE=%0.3f',N,M,MSE),'FontSize',15)
view([0 0 1])
colorbar
xlabel('x_1')
ylabel('x_2')



% Sparse MDCE
py(:) = sparsemdce([px1(:),px2(:)]);
MSEs = mean((py0(:) - py(:)).^2);
nexttile
surf(px1,px2,py,'EdgeColor','none')
title('MDCE estimated by Sparse GPR',sprintf('N=%d, M=%d, MSE=%0.3f',N,M,MSEs),'FontSize',15)
% zlim(1.1*mdcelim)
clim([-10,10]);
% zlim([-10,10])
view([0 0 1])
colorbar
xlabel('x_1')
ylabel('x_2')



%% Save the plots
set(gcf,'Position',[10 97 521 683]);
saveas(gcf,'./Results/sparsity.png')
saveas(gcf,'./Results/sparsity.svg')

%% Functions


function d2Fdxdz = getderivative(gp,xp,x)
    alpha = gp.Alpha;
    if strcmp(gp.KernelInformation.Name,'SquaredExponential')
        l = gp.KernelInformation.KernelParameters(1);
        sf= gp.KernelInformation.KernelParameters(end);
        dx1 = (xp(:,1)-x(:,1)')/(-l^2);
        dx2 = (xp(:,2)-x(:,2)')/(-l^2);
        del12 = 1/(-l*l);
        ddkdxdx = (sf.^2)*exp(-0.5*(pdist2(xp./l',x./l')).^2).*(dx1.*dx2 );
    elseif strcmp(gp.KernelInformation.Name,'ARDSquaredExponential')
        l = gp.KernelInformation.KernelParameters(1:end-1);
        sf= gp.KernelInformation.KernelParameters(end);
        dx1 = (xp(:,1)-x(:,1)')/(-l(1)^2);
        dx2 = (xp(:,2)-x(:,2)')/(-l(2)^2);
        del12 = 1/(-l(1)*l(2));
        ddkdxdx = (sf.^2)*exp(-0.5*(pdist2(xp./l',x./l')).^2).*(dx1.*dx2 );
    elseif strcmp(gp.KernelInformation.Name,'Matern52')
        l = gp.KernelInformation.KernelParameters(1);
        sf= gp.KernelInformation.KernelParameters(2);
        dx1 = (xp(:,1)-x(:,1)');
        dx2 = (xp(:,2)-x(:,2)');
%         del12 = 1/(-l(1)*l(2));
%         ddkdxdx = (sf.^2)*exp(-0.5*(pdist2(xp./l',x./l')).^2).*(dx1.*dx2 );
        r = pdist2(xp./l',x./l');
        drdx1 = dx1./r;
        drdx2 = dx2./r;
        ddrdxdx= -dx1.*dx2./(r.^3);
        dkdr = (sf^2)*exp(-sqrt(5)*r/l).*(-5*sqrt(5)*r.^2/(3*l^3) + -5*r/(3*l^2));
        ddkdr2 =  (sf^2)*exp(-sqrt(5)*r/l).*(25*r.^2/(3*l^4) - 5*sqrt(5)*r/(3*l^3) - 5/(3*l^2));
        ddkdxdx = ddkdr2.*drdx1.*drdx2 + dkdr.*ddrdxdx;
    elseif strcmp(gp.KernelInformation.Name,'CustomKernel')
        l3 = (exp(gp.KernelInformation.KernelParameters(6)));
        l4 = (exp(gp.KernelInformation.KernelParameters(7)));
        l34 = [l3 l4];
        var12= exp(gp.KernelInformation.KernelParameters(3));
        dx1 = (xp(:,1)-x(:,1)')/(-l3^2);
        dx2 = (xp(:,2)-x(:,2)')/(-l4^2);
        del12 = 1/(-l3*l4);
        ddkdxdx = var12*exp(-0.5*(pdist2(xp(:,1),x(:,1))/l3).^2-0.5*(pdist2(xp(:,2),x(:,2))/l4).^2).*(dx1.*dx2 );
    end
    d2Fdxdz = ddkdxdx*alpha;
end

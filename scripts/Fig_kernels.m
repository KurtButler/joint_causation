%% Figure - Comparison of Kernels
% This script should produce a figure that compares the MDCE estimates for
% each of the kernels for the 


disp(' <> <> <> <> Egg Box <> <> <> <>')

func1 = @(x) 0.5*cos(3*x(:,1))+0.5*cos(3*x(:,2));
func2 = @(x) cos(3*x(:,1)).*cos(2*x(:,2));
% func2 = @(x) 0.5*cos(3*x(:,1)+3*x(:,2))+0.5*cos(3*x(:,2)-3*x(:,1));
func3 = @(x) sin(2*x(:,1)).*sin(4*x(:,2));% + cos(2*x(:,1))+cos(2*x(:,2));



% Test Case 1: Egg box
% func3 = @(x) sin(x(:,1)) + sin(x(:,2));

% Test Case 2: Modified egg box
% func3 = @(x) sin(2*x(:,1)).*sin(2*x(:,2));

% Test Case 3: Region-specific interactions
sigmoid = @(x) 1./(1+exp(-5*x));
func3 = @(x) cos(2*x(:,1)).*cos(3*x(:,2)).*sigmoid(x(:,1)) + sin(x(:,2));

% func3 = @(x) cos(2*x(:,1)).*cos(2*x(:,2));
% func3 = @(x) x(:,1).*x(:,2);

% Test Case 4: 
% func3 = @(x) sin(2*x(:,1)).*sin(2*x(:,2)) + 4*x(:,1).*x(:,2);


SNR = 5;

ddfunc3 = @(x) 1e8*(func3(x-[1e-4 1e-4]) - func3(x-[0 1e-4])-func3(x-[1e-4 0]) + func3(x));


[demox1,demox2] = meshgrid(linspace(-2,2,50));
demoy1 = zeros(size(demox1));
demoy2 = zeros(size(demox1));
demoy3 = zeros(size(demox1));
demoy1(:) = func1([demox1(:), demox2(:)]);
demoy2(:) = func2([demox1(:), demox2(:)]);
demoy3(:) = func3([demox1(:), demox2(:)]);



% Signal power
NOISEPOWER = nan;
fprintf('Signal power: %0.3f\n',mean(func3([demox1(:),demox2(:)]).^2,[1,2]));
fprintf('MDCE   power: %0.3f\n',mean(ddfunc3([demox1(:),demox2(:)]).^2,[1,2]));
fprintf('Noise power:  %0.3f\n',NOISEPOWER);



%% Data generative process
N = 300;
x = 4*rand(N,2) - 2;
fy = func3(x);
y = awgn(fy,SNR,'measured');

%% GP models
gp1 = fitrgp(x,y,'KernelFunction','squaredexponential');
gp2 = fitrgp(x,y,'KernelFunction','ardsquaredexponential');

kfcn = @(XN,XM,theta) addkernel(XN,XM,theta);
theta0 = [0 0 0 0 0 0 0]';
gp3 = fitrgp(x,y,'KernelParameters',theta0,'KernelFunction',kfcn);

gp4 = fitrgp(x,y,'KernelFunction','matern52');

kfcn = @(XN,XM,theta) perkernel(XN,XM,theta);
theta0 = [0 3 3 1 1]';
gp5 = fitrgp(x,y,'KernelParameters',theta0,'KernelFunction',kfcn);

xp  = 4*rand(99,2) - 2;
yp0 = func3(xp);
yp1 = predict(gp1,xp);
yp2 = predict(gp2,xp);
yp3 = predict(gp3,xp);
yp4 = predict(gp4,xp);
yp5 = predict(gp5,xp);

%% Plot results
% figure(2)
% tiledlayout(2,2,'TileSpacing','compact','Padding','tight')
% 
% nexttile
% surf(demox1,demox2,demoy3,'FaceAlpha',0.95,'EdgeAlpha',0.0)
% colormap(gray)
% hold on;
% plot3(x(:,1),x(:,2),y,'k*',xp(:,1),xp(:,2),yp1,'o',xp(:,1),xp(:,2),yp2,'x',xp(:,1),xp(:,2),yp3,'+');
% hold off;
% legend('','Observed','SE','ARD-SE','Additive','FontSize',15)
% 
% nexttile
% plot(yp0,yp1,'o',yp0,yp2,'x',yp0,yp3,'+',[-2;2],[-2;2],'k--'); 
% xlabel('Function true value'); 
% ylabel('GP estimate');
% title('Predictions of FUNCTION vs true values','FontSize',15)
% legend('SE','ARD-SE','Additive','FontSize',15)
% 
% nexttile
% err1 = (yp0 - yp1).^2;
% err2 = (yp0 - yp2).^2;
% err3 = (yp0 - yp3).^2;
% histogram(err1);
% hold on;
% histogram(err2);
% histogram(err3);
% hold off;
% title('Histograms of error','FontSize',15)
% legend('SE','ARD-SE','Additive','FontSize',15)
% 
% % Text output
% fprintf('= = = MSE of function = = = \n')
% fprintf('  SE\tARD-SE\tAdd.\n')
% fprintf('  %0.3f\t%0.3f\t%0.3f\n',mean(err1),mean(err2),mean(err3))


%% Plot derivatives
% figure(3)
% tiledlayout(1,1,'TileSpacing','compact','Padding','tight')
% 
% MDCE1 = getderivative(gp1,xp,x);
% MDCE2 = getderivative(gp2,xp,x);
% MDCE3 = getderivative(gp3,xp,x);
% 
% surface = zeros(size(demox1));
% surface(:) = ddfunc3([demox1(:),demox2(:)]);
% 
% nexttile
% surf(demox1,demox2,surface,'FaceAlpha',0.95,'EdgeAlpha',0.0)
% colormap(gray)
% % hold on;
% % plot3(x(:,1),x(:,2),ddfunc3(x),'k*',xp(:,1),xp(:,2),MDCE1,'o',xp(:,1),xp(:,2),MDCE2,'x',xp(:,1),xp(:,2),MDCE3,'+');
% % hold off;
% % legend('','Observed','SE','ARD-SE','Additive','FontSize',15)
% title('MDCE','FontSize',15)
% 
% 

%% Plot function
POINTSIZE = 10;

figure(1)
tiledlayout(1,2,'TileSpacing','compact','Padding','tight')

% nexttile
% surf(demox1,demox2,demoy1)
% title('Eggbox','FontSize',15)
% 
% nexttile
% surf(demox1,demox2,demoy2)
% title('Modified Eggbox','FontSize',15)

nexttile
surf(demox1,demox2,demoy3,'EdgeColor','black','EdgeAlpha',0.2)
hold on;
scatter3(x(:,1),x(:,2), y,POINTSIZE,'black','filled')
for n = 1:size(x,1)
    plot3(x(n,1)*[1;1],x(n,2)*[1;1], [func3([x(n,:)]);y(n)],'k');
end
hold off;
title('Local interaction function, F','FontSize',15)
colorbar
xlabel('x_1')
ylabel('x_2')
zlabel('y')

nexttile
ddemoy = demoy3;
ddemoy(:) = ddfunc3([demox1(:), demox2(:)]);
surf(demox1,demox2,ddemoy,'EdgeColor','black','EdgeAlpha',0.2)
title('Mixed-derivative, \partial_1\partial_2F','FontSize',15)
colorbar
xlabel('x_1')
ylabel('x_2')
zlabel('\partial_1\partial_2y')

% uu = (1/256)*linspace(-150,256,500)';
% redblue = max([uu,flipud([uu,uu])],0);
% colormap(redblue)

% saveas(gcf,'./Results/local_function.png')
% saveas(gcf,'./Results/local_function.svg')



%% Compute MDCE
MDCE0 = zeros(size(demox1));
MDCE1 = zeros(size(demox1));
MDCE2 = zeros(size(demox1));
MDCE3 = zeros(size(demox1));
MDCE4 = zeros(size(demox1));
MDCE5 = zeros(size(demox1));

MDCE0(:) = ddfunc3([demox1(:),demox2(:)]);
MDCE1(:) = getderivative(gp1,[demox1(:),demox2(:)],x);
MDCE2(:) = getderivative(gp2,[demox1(:),demox2(:)],x);
MDCE3(:) = getderivative(gp3,[demox1(:),demox2(:)],x);
MDCE4(:) = getderivative(gp4,[demox1(:),demox2(:)],x);
MDCE5(:) = perderivative(gp5,[demox1(:),demox2(:)],x);


gpnumdiff = @(gp,x) 1e8*(predict(gp,x-[1e-4 1e-4]) - predict(gp,x-[0 1e-4])-predict(gp,x-[1e-4 0]) + predict(gp,x));



% MDCE4(:) = gpnumdiff(gp4,[demox1(:),demox2(:)]);
% MDCE5(:) = gpnumdiff(gp5,[demox1(:),demox2(:)]);


%% Debugger for Periodic kernel 
% POINTSIZE = 10;
% 
% MDCEperanalytic = zeros(size(demox1));
% MDCEpernumerical = zeros(size(demox1));
% MDCEperanalytic(:) = perderivative(gp5,[demox1(:),demox2(:)],x);
% MDCEpernumerical(:) = gpnumdiff(gp5,[demox1(:),demox2(:)]);
% 
% 
%  
% figure(24)
% tiledlayout(2,2,'TileSpacing','compact','Padding','tight')
% 
% 

% nexttile
% surf(demox1,demox2,MDCE0,'EdgeColor','black','EdgeAlpha',0)
% title('True MDCE',sprintf('SNR=%ddB',SNR),'FontSize',15)
% hold on;
% scatter3(x(:,1),x(:,2), max(MDCE0(:))+1 + 0*x(:,1),POINTSIZE,'black','filled')
% hold off;
% legend('','Sample point locations','Location','northwest')
% clim([-1,1]*limiter)
% view([0 0 1])
% colorbar
% 
% nexttile
% surf(demox1,demox2,MDCE1,'EdgeColor','black','EdgeAlpha',0)
% title('SE kernel','FontSize',15)
% clim([-1,1]*limiter)
% view([0 0 1])
% colorbar
% 
% 
% nexttile
% surf(demox1,demox2,MDCEperanalytic,'EdgeColor','black','EdgeAlpha',0)
% title('Periodic kernel','(analytic)','FontSize',15)
% clim([-1,1]*limiter)
% view([0 0 1])
% colorbar
% 
% 
% 
% nexttile
% surf(demox1,demox2,MDCEpernumerical,'EdgeColor','black','EdgeAlpha',0)
% title('Periodic kernel','(numerical)','FontSize',15)
% clim([-1,1]*limiter)
% view([0 0 1])
% colorbar
% 
% colormap(turbo)



%% Text output
mdceerr1 = (MDCE1-MDCE0).^2;
mdceerr2 = (MDCE2-MDCE0).^2;
mdceerr3 = (MDCE3-MDCE0).^2;
mdceerr4 = (MDCE4-MDCE0).^2;
mdceerr5 = (MDCE5-MDCE0).^2;

ModelEnergy = mean(MDCE0.^2,'all');
ResidualEnergy1 = mean(mdceerr1,'all');
ResidualEnergy2 = mean(mdceerr2,'all');
ResidualEnergy3 = mean(mdceerr3,'all');
ResidualEnergy4 = mean(mdceerr4,'all');
ResidualEnergy5 = mean(mdceerr5,'all');

fprintf('= = = MSE of MDCE = = = \n')
fprintf('  SE\tARD-SE\tAdd.\tMat52\tPer\n')
fprintf('  %0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n',ResidualEnergy1,ResidualEnergy2,...
    ResidualEnergy3,ResidualEnergy4,ResidualEnergy5)


R2model1 = 1 - ResidualEnergy1/ModelEnergy;
R2model2 = 1 - ResidualEnergy2/ModelEnergy;
R2model3 = 1 - ResidualEnergy3/ModelEnergy;
R2model4 = 1 - ResidualEnergy4/ModelEnergy;
R2model5 = 1 - ResidualEnergy5/ModelEnergy;

%% Plot heatmaps of derivatives
POINTSIZE = 10;

figure(4)
tiledlayout(3,2,'TileSpacing','compact','Padding','tight')


limiter = max(abs([MDCE0(:);MDCE1(:);MDCE2(:);MDCE3(:)]));

nexttile
surf(demox1,demox2,MDCE0,'EdgeColor','black','EdgeAlpha',0)
title('True MDCE',sprintf('SNR=%ddB',SNR),'FontSize',15)
hold on;
scatter3(x(:,1),x(:,2), max(MDCE0(:))+1 + 0*x(:,1),POINTSIZE,'black','filled')
hold off;
legend('','Sample point locations','Location','northwest')
clim([-1,1]*limiter)
view([0 0 1])
colorbar

nexttile
surf(demox1,demox2,MDCE1,'EdgeColor','black','EdgeAlpha',0)
title('SE kernel',sprintf('MSE=%0.3f',ResidualEnergy1),'FontSize',15)
clim([-1,1]*limiter)
view([0 0 1])
colorbar

nexttile
surf(demox1,demox2,MDCE2,'EdgeColor','black','EdgeAlpha',0)
title('ARD-SE kernel',sprintf('MSE=%0.3f',ResidualEnergy2),'FontSize',15)
clim([-1,1]*limiter)
view([0 0 1])
colorbar

nexttile
surf(demox1,demox2,MDCE3,'EdgeColor','black','EdgeAlpha',0)
title('Additive kernel',sprintf('MSE=%0.3f',ResidualEnergy3),'FontSize',15)
clim([-1,1]*limiter)
view([0 0 1])
colorbar

nexttile
surf(demox1,demox2,MDCE4,'EdgeColor','black','EdgeAlpha',0)
title('Matern 5/2 kernel',sprintf('MSE=%0.3f',ResidualEnergy4),'FontSize',15)
% clim([-1,1]*limiter)
view([0 0 1])
colorbar

nexttile
surf(demox1,demox2,MDCE5,'EdgeColor','black','EdgeAlpha',0)
title('Periodic kernel',sprintf('MSE=%0.3f',ResidualEnergy5),'FontSize',15)
% clim([-1,1]*limiter)
view([0 0 1])
colorbar


% sgtitle('MDCE predictions for various kernels','FontSize',19)


%% Save the result
% saveas(gcf,'./Results/kernel_comparison.png')


%% Functions
function K = addkernel(Xm,Xn,theta)
    var1 = exp(theta(1));
    var2 = exp(theta(2));
    var12 = exp(theta(3));
    l1 = exp(theta(4));
    l2 = exp(theta(5));
    l3 = exp(theta(6));
    l4 = exp(theta(7));
    K = var1*exp(-0.5*(pdist2(Xm(:,1),Xn(:,1))/l1).^2) ...
        + var2*exp(-0.5*(pdist2(Xm(:,2),Xn(:,2))/l2).^2) ...
        + var12*exp(-0.5*(pdist2(Xm(:,1),Xn(:,1))/l3).^2-0.5*(pdist2(Xm(:,2),Xn(:,2))/l4).^2);
end


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
        r = pdist2(xp,x);
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


% Product of two periodic kernels
function K = perkernel(Xm,Xn,theta)
    var12 = exp(theta(1));
    l1 = exp(theta(2));
    l2 = exp(theta(3));
    f1 = theta(4);
    f2 = theta(5);
    % To save my sanity, I'm going to code this as a product of dimension 1
    % kernels
    distmat1 = pdist2(Xm(:,1),Xn(:,1));  % Distances in coordinate 1
    distmat2 = pdist2(Xm(:,2),Xn(:,2));  % Distances in coordinate 2
    K1 = exp(-(sin(f1*distmat1)/l1).^2);
    K2 = exp(-(sin(f2*distmat2)/l2).^2);
    K = var12*K1.*K2;
end

function d2Fdxdz = perderivative(gp,xp,x)
alpha = gp.Alpha;
theta = gp.KernelInformation.KernelParameters;
var12 = exp(theta(1));
l1 = exp(theta(2));
l2 = exp(theta(3));
f1 = theta(4);
f2 = theta(5);
% I will now exploit the product factorization to make this analysis less
% awful.
distmat1 = (xp(:,1)-x(:,1)');  % DIFFERENCES in coordinate 1
distmat2 = (xp(:,2)-x(:,2)');  % DIFFERENCES in coordinate 2
K1 = exp(-(sin(f1*distmat1)/l1).^2);
K2 = exp(-(sin(f2*distmat2)/l2).^2);
dx1 = f1*sin(f1*distmat1).*cos(f1*distmat1)/(-l1^2);
dx2 = f2*sin(f2*distmat2).*cos(f2*distmat2)/(-l2^2);
ddkdxdx = 4*var12*K1.*K2.*dx1.*dx2;
d2Fdxdz = ddkdxdx*alpha;
end

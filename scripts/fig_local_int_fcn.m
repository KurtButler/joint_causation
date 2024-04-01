%% Figure - Comparison of Kernels
% This script should produce a figure that compares the MDCE estimates for
% each of the kernels for the 

% Define the local interaction function
sigmoid = @(x) 1./(1+exp(-5*x));
func3 = @(x) cos(2*x(:,1)).*cos(3*x(:,2)).*sigmoid(x(:,1)) + sin(x(:,2));

SNR = 5;

ddfunc3 = @(x) 1e8*(func3(x-[1e-4 1e-4]) - func3(x-[0 1e-4])-func3(x-[1e-4 0]) + func3(x));


[demox1,demox2] = meshgrid(linspace(-2,2,50));
demoy3 = zeros(size(demox1));
demoy3(:) = func3([demox1(:), demox2(:)]);

%% Data generative process
N = 300;
x = 4*rand(N,2) - 2;
fy = func3(x);
y = awgn(fy,SNR,'measured');

%% Plot function
POINTSIZE = 10;

figure(1)
tiledlayout(1,2,'TileSpacing','compact','Padding','tight')

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


%% Save the result
set(gcf,'Position', [23 146 605 233] )
saveas(gcf,'./results/fig1.png')


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

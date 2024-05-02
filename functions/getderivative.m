function d2Fdxdz = getderivative(gp,xp,x,CustomKernelName)
%GETDERIVATIVE Summary of this function goes here
%   Detailed explanation goes here
if nargin < 4
    CustomKernelName = 'none';
end

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


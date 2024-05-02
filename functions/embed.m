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
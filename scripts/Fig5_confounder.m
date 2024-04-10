%% Control for the random seed
rng(0)

%% Major parameters
Dx = 3;
NoTrials = 300; % No. of trials
Ntrain = 500; % No. of training points per trial
Ntest  = 100; % No. of test points per trial

%% Simulation parameters
slist = [0 0.2 0.4 0.6 0.8 1];
slist = linspace(0,1,20);
sigx = 2;
sigy = 1;
bilincoeff = 0.5;
bilinbase = 1;

%% Initialize our experiment matrices
MSEmatrix = zeros(NoTrials,2); % there
outputTensor = zeros(numel(slist), NoTrials,2);

%% Main loop
for iter = 1:numel(slist)
    zsigma = slist(iter);
for testcase = 1:2
    fprintf('Iter %d/%d, Case %d/2\n',iter,numel(slist),testcase)
    for trial = 1:NoTrials
        %% Define our functions
        % x := g(z) + w_x
        % y := f(x,z) + w_y
        Bmat = randn(Dx).*(rand(Dx)>0.5).*((1:Dx)>(1:Dx)');
        while   all(Bmat==0)
            Bmat = randn(Dx).*(rand(Dx)>0.5).*((1:Dx)>(1:Dx)');
        end

        if testcase == 1
            Gfcn = @(z)   z*rand(1,Dx);
            Ffcn = @(x,z) diag(x*Bmat*x') + 5*z + x*ones(Dx,1);
        else
            Gfcn = @(z)   z*rand(1,Dx);
            Ffcn = @(x,z) diag(x*Bmat*x').*(bilincoeff*z + bilinbase) + 5*z + x*ones(Dx,1);
        end

        %% Generate data
        N = Ntrain + Ntest;
        z = zsigma*randn(N,1);
        x = Gfcn(z)   + sigx*randn(N,1);
        y = Ffcn(x,z) + sigy*randn(N,1);

        %% Partition the data
        xtrain = x(1:Ntrain,:);
        xtest = x(Ntrain+1:Ntrain+Ntest,:);
        ytrain = y(1:Ntrain,:);
        ytest  = y(Ntrain+1:Ntrain+Ntest,:);
        %         ztrain = z(1:Ntrain,:); % unused
        ztest  = z(Ntrain+1:Ntrain+Ntest,:); % we will use this in testing our accuracy

        %% Train the model
        gp = fitrgp(xtrain,ytrain,'KernelFunction','ardsquaredexponential',...
            'BasisFunction','constant');

        %% Compute the true Hessian matrix
        HessTrue = zeros(Ntest,Dx^2);
        if testcase == 1
            % Linear confounders
            for i1=1:Dx
                for i2=1:Dx
                    % Expression for the bilinear model:
                    HessTrue(:, Dx*(i1-1) + i2) = Bmat(i1,i2) + Bmat(i2,i1);
                end
            end
        else
            % Evil confounders
            for i1=1:Dx
                for i2=1:Dx
                    % Expression for the bilinear model:
                    HessTrue(:, Dx*(i1-1) + i2) = Bmat(i1,i2) + Bmat(i2,i1);
                end
            end
            HessTrue = (bilincoeff*ztest+bilinbase).*HessTrue;
        end

        %% Estimate the Hessian via GPR
        HessPred = zeros(Ntest,Dx^2);
        ddyp = zeros(Ntest,1);
        for i1=1:Dx
            for i2=i1:Dx
                % Formulas for the ARD-SE kernel
                alpha = gp.Alpha;
                l = gp.KernelInformation.KernelParameters(1:end-1);
                sf= gp.KernelInformation.KernelParameters(end);
                sg= gp.Sigma;

                dx1 = (xtest(:,i1)-xtrain(:,i1)')/(-l(i1)^2);
                dx2 = (xtest(:,i2)-xtrain(:,i2)')/(-l(i2)^2);
                del12 = 1/(-l(i1)*l(i2));

                dkdx = (sf.^2)*exp(-0.5*(pdist2(xtest./l',xtrain./l')).^2).*dx1;
                ddkdxdx = (sf.^2)*exp(-0.5*(pdist2(xtest./l',xtrain./l')).^2).*(dx1.*dx2 + del12);

                dyp = dkdx*alpha;
                ddyp(:) = ddkdxdx*alpha;

                HessPred(:,i1 + Dx*(i2-1)) = ddyp(:);
                HessPred(:,i2 + Dx*(i1-1)) = ddyp(:);
            end
        end

        HessErrors = HessPred(:) - HessTrue(:);
        myMSE = mean(HessErrors.^2);

        %% Save the MSE
        MSEmatrix(trial,testcase) = myMSE;
    end
end
outputTensor(iter,:,:) = MSEmatrix;
end


%% Plot some results
figure(18)
tiledlayout(1,1,'Padding','tight')

nexttile
fsummarize = @(X) median(X,2);
for testcase = 1:2
    if testcase == 1
        colorstring = 'bo-';
    else
        colorstring = 'ro-';
    end
    plot(slist, fsummarize(outputTensor(:,:,testcase)),colorstring)
    hold on;
end
hold off;
grid on;
grid minor;
legend('Linear confounders','General confounder','Location','best','FontSize',10)
title('MDCE estimation error vs confounder strength','FontSize',15)
xlabel('Confounder strength coefficient, \sigma_z')
ylabel('MDCE estimation error')


%% Export
set(gcf,'Position',[56 261 560 177]);
saveas(gcf,'./results/confounder.png')


clc;clear;
disp("Start of script");
% Load data - Split data
data=load('airfoil_self_noise.dat');
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);


% Evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);


% FIS with grid partition
fis(1) = genfis1(trnData,2,'gbellmf','constant');
fis(2) = genfis1(trnData,3,'gbellmf','constant');
fis(3) = genfis1(trnData,2,'gbellmf','linear');
fis(4) = genfis1(trnData,3,'gbellmf','linear');


for i = 1:2
    % ANFIS
    [trnFis,trnError,~,valFis,valError] = anfis(trnData,fis(i),[100 0 0.01 0.9 1.1],[],chkData); % 100 epochs
    
    % Membership functions plots
    message = "TSK model ";
    numberTSK = int2str(i);
    message = strcat(message,numberTSK);
    message = strcat(message," membership functions for input ");
    for j = 1:size(trnData,2)-1
        input = int2str(j);
        Title = strcat(message,input);
        figure(i*100 + j);
        plotmf(valFis,'input',j);
        title(Title);
    end

    % Learning Curve plot
    figure(i*100 + 10 + i);
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    message = "TSK model ";
    message = strcat(message,numberTSK);
    message = strcat(message," learning curve");
    title(message);
    Y = evalfis(tstData(:,1:end-1),valFis);
    R2 = Rsq(Y,tstData(:,end));
    RMSE = sqrt(mse(Y,tstData(:,end)));
    NMSE = 1 - R2;
    NDEI = sqrt(NMSE);
    Perf(i,:) = [R2; RMSE; NMSE; NDEI];
    
    % Prediction Error plot
    predictionError = tstData(:,end) - Y;
    figure(100*i + 20);
    plot(predictionError,'LineWidth',2); grid on;
    message = "TSK model ";
    message = strcat(message,numberTSK);
    message = strcat(message," prediction error");
    xlabel('input');ylabel('Error');
    title(message);
end

% Results Table
varnames={'TSK_model_1','TSK_model_2','TSK_model_3','TSK_model_4'};
rownames={'Rsquared','RMSE','NMSE','NDEI'};
Perf = Perf';
Perf = array2table(Perf,'VariableNames',varnames,'RowNames',rownames)
disp("End of script");
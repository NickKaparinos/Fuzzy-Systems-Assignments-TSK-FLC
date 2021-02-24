clc;clear;
disp("Start of script");

% Read data and normalise
data = csvread('train.csv',1,0);
data = normaliseData(data);

Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% Grid search
a = 5;          
b = 7;
% First param is num of features, second param is clusters ra
gridSearchParams = zeros(a,b,2);

gridSearchParams(1,:,1) = 5;
gridSearchParams(2,:,1) = 10;
gridSearchParams(3,:,1) = 15;
gridSearchParams(4,:,1) = 20;
gridSearchParams(5,:,1) = 25;
gridSearchParams(:,1,2) = 0.2;
gridSearchParams(:,2,2) = 0.3;
gridSearchParams(:,3,2) = 0.4;
gridSearchParams(:,4,2) = 0.5;
gridSearchParams(:,5,2) = 0.6;
gridSearchParams(:,6,2) = 0.7;
gridSearchParams(:,7,2) = 0.8;

errors = zeros(a,b,2);

% k-fold cross validation is used
k = 5;

tic
% Grid search
for i = 1:a
    for j = 1:b
        numOfFeatures = gridSearchParams(i,j,1);
        ra = gridSearchParams(i,j,2);
        
        crossValErrors = zeros(k,2);
        
        % k-fold cross validation
        for iteration = 1:k
            [trnData,chkData,tstData] = crossValidationDatasets(k,data,iteration);
            
            % Feature selection
            [idx,weights] = relieff( trnData(:,1:end-1), trnData(:,end),5);
            
            %idx = idx( 1:numOfFeatures );
            trnDataFS = trnData( :, idx(1:numOfFeatures) );
            trnDataFS = [ trnDataFS trnData( :, end)];
            
            chkDataFS = chkData( :, idx(1:numOfFeatures) );
            chkDataFS = [ chkDataFS chkData( :, end) ];
            
            tstDataFS = tstData( :, idx(1:numOfFeatures) );
            tstDataFS = [ tstDataFS tstData( :, end) ];
            
            
            % genfis2 (SC)
            fis = genfis2(trnDataFS(:,1:end-1),trnDataFS(:,end),ra);
            
            
            % Training
            disp("Start of Training");
            [trnFis,trnError,~,valFis,valError] = anfis(trnDataFS,fis,[100 0 0.01 0.9 1.1],[],chkDataFS);
            figure(iteration);
            plot([trnError valError],'LineWidth',2); grid on;
            xlabel('# of Iterations'); ylabel('Error');
            legend('Training Error','Validation Error');
            title('ANFIS Hybrid Training - Validation');
            disp("End of Training");
            
            Y = evalfis(tstDataFS(:,1:end-1),valFis);
            R2 = Rsq(Y,tstDataFS(:,end));
            RMSE = sqrt(mse(Y,tstDataFS(:,end)));
            NMSE = 1 - R2;
            NDEI = sqrt(NMSE);
            disp("End of Training");
          
            % Save errors for cross validation
            crossValErrors(iteration,1) = R2;
            crossValErrors(iteration,2) = RMSE;
        end
        % Find average of cross validation errors and save it
        tempErrorR2 = sum( crossValErrors(:,1) ) / k;
        tempErrorRMSE = sum( crossValErrors(:,2) ) / k;
        
        errors(i,j,1) = tempErrorR2;
        errors(i,j,2) = tempErrorRMSE;
        
        
    end
end
toc

errors
disp("End of script");
%clc;clear;
disp("Start of script");

% Read data and normalise
data = csvread('data.csv',1,1);
data = normaliseData(data);

% Optimal Model
numOfFeatures = 10;
ra = 0.5;

% k-fold cross validaiton
k = 5;

% Crossvalidate R2 and RMSE
crossValErrors = zeros(k,2);

% k-fold cross validation
crossValOA = zeros(k,1);
crossValPA = zeros(5,k);
crossValUA = zeros(5,k);
crossValk  = zeros(k,1);
crossValErrorMatrix = zeros(5,5,5);
cvPart = cvpartition(data(:,end),'KFold',5,'Stratify',true);

% k-fold cross validation
for iteration = 1:k
    trnDataTemp = data(training(cvPart,iteration),:);
    tstData = data(test(cvPart,iteration),:);
    % cv partition is used to split trnDataTemp into trnData and
    % chkData using stratification
    cvPartitionTrn = cvpartition(trnDataTemp(:,end),'KFold',4,'Stratify',true);
    trnData = trnDataTemp(training(cvPartitionTrn,1), :);
    chkData = trnDataTemp(test(cvPartitionTrn,1), :);
    
    % Feature selection
    [idx,weights] = relieff( trnData(:,1:end-1), trnData(:,end),5);
    
    trnDataFS = trnData( :, idx(1:numOfFeatures) );
    trnDataFS = [ trnDataFS trnData( :, end)];
    
    chkDataFS = chkData( :, idx(1:numOfFeatures) );
    chkDataFS = [ chkDataFS chkData( :, end) ];
    
    tstDataFS = tstData( :, idx(1:numOfFeatures) );
    tstDataFS = [ tstDataFS tstData( :, end) ];
    
    % Clustering Per Class
    [c1,sig1] = subclust(trnDataFS(trnDataFS(:,end) == 1,:),ra);
    [c2,sig2] = subclust(trnDataFS(trnDataFS(:,end) == 2,:),ra);
    [c3,sig3] = subclust(trnDataFS(trnDataFS(:,end) == 3,:),ra);
    [c4,sig4] = subclust(trnDataFS(trnDataFS(:,end) == 4,:),ra);
    [c5,sig5] = subclust(trnDataFS(trnDataFS(:,end) == 5,:),ra);
    num_rules = size(c1,1) + size(c2,1) + size(c3,1) + size(c4,1) + size(c5,1);
    
    % Build FIS From Scratch
    fis2 = newfis('FIS_SC','sugeno');
    
    % Add Input-Output Variables
    names_in = {};
    for i = 1:size(trnDataFS,2)-1
        num = int2str(i);
        name = 'input';
        name = strcat(name,num);
        names_in = [names_in name];
    end
    for i = 1:size(trnDataFS,2)-1
        fis2 = addvar(fis2,'input',names_in{i},[0 1]);
    end
    fis2 = addvar(fis2,'output','out1',[0 1]);
    
    % Add Input Membership Functions
    name = 'sth';
    for i = 1:size(trnDataFS,2)-1
        for j=1:size(c1,1)
            fis2 = addmf(fis2,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]);
        end
        for j=1:size(c2,1)
            fis2 = addmf(fis2,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
        end
        for j=1:size(c3,1)
            fis2 = addmf(fis2,'input',i,name,'gaussmf',[sig3(i) c3(j,i)]);
        end
        for j=1:size(c4,1)
            fis2 = addmf(fis2,'input',i,name,'gaussmf',[sig4(i) c4(j,i)]);
        end
        for j=1:size(c5,1)
            fis2 = addmf(fis2,'input',i,name,'gaussmf',[sig5(i) c5(j,i)]);
        end
    end
    
    temp2 = zeros(1,size(c2,1));
    temp3 = zeros(1,size(c3,1));
    temp4 = zeros(1,size(c4,1));
    temp2 = temp2 + 0.25;
    temp3 = temp3 + 0.5;
    temp4 = temp4 + 0.75;
    % Add Output Membership Functions
    params = [zeros(1,size(c1,1)) temp2 temp3 temp4 ones(1,size(c5,1))];
    for i = 1:num_rules
        fis2 = addmf(fis2,'output',1,name,'constant',params(i));
    end
    
    % Add FIS Rule Base
    ruleList = zeros(num_rules,size(trnDataFS,2));
    for i = 1:size(ruleList,1)
        ruleList(i,:)=i;
    end
    ruleList = [ruleList ones(num_rules,2)];
    fis2 = addrule(fis2,ruleList);
    
    % Plot mf before training
    titleBefore = "Optimal TSK model membership functions for input ";
    figure(1);
    plotmf(fis2,'input',1);
    title1 = strcat(titleBefore,'1');
    title1 = strcat(title1,' before training');
    title(title1);
    
    figure(2);
    plotmf(fis2,'input',size(trnDataFS,2)-1);
    num = int2str(size(trnDataFS,2)-1);
    title2 = strcat(titleBefore,num);
    title2 = strcat(title2,' before training');
    title(title2);
    
    % Train & Evaluate ANFIS
    [trnFis,trnError,~,valFis,valError]=anfis(trnDataFS,fis2,[150 0 0.01 0.9 1.1],[],chkDataFS);
    
    % Learning curve plot
    figure(1000);
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    Y=evalfis(tstDataFS(:,1:end-1),valFis);
    Y=round(Y);
    diff=tstDataFS(:,end)-Y;
    Acc=(length(diff)-nnz(diff))/length(Y)*100;
    
    % Plot mf after training
    titleAfter = "Optimal TSK model membership functions for input ";
    figure(4);
    plotmf(valFis,'input',1);
    title1 = strcat(titleAfter,'1');
    title1 = strcat(title1,' after training');
    title(title1);
    
    figure(5);
    plotmf(valFis,'input',size(trnDataFS,2)-1);
    num = int2str(size(trnDataFS,2)-1);
    title2 = strcat(titleAfter,num);
    title2 = strcat(title2,' after training');
    title(title2);
    
    % Predictions plot
    figure(7)
    scatter(1:size(Y,1),Y,'.'); grid on;
    xlabel('input');
    legend('Prediction');
    title('Model predictions');
    
    % Ground truth
    figure(8)
    scatter(1:size(tstDataFS,1),tstDataFS(:,end),'.');grid on;
    xlabel('input');
    legend('Ground truth');
    title('Ground truth');
    
    % Prediction Error plot
    predictionError = tstDataFS(:,end) - Y;
    figure(9);
    scatter(1:size(predictionError,1),predictionError,'.'); grid on;
    xlabel('input');ylabel('Error');
    legend('Prediction Error');
    title('Optimal TSK model prediction error')
    
    classes = [1 2 3 4 5];
    errorMatrix = zeros(5);
    N = size(tstDataFS,1);
    % Error matrix
    for i = 1:5
        for j = 1:5
            errorMatrix(i,j) = size( intersect( find( Y == classes(i) ) , find(tstData(:,end) == classes(j) ) ) ,1);
        end
    end
    
    % OA
    sumCorrect = trace(errorMatrix);
    OA = 1/N*sumCorrect;
    
    % Producers accuracy and users accuracy
    sumRows = zeros(5,1);
    sumColumns = zeros(5,1);
    PA = zeros(5,1);
    UA = zeros(5,1);
    for i = 1:5
        sumRows(i) = sum( errorMatrix(i,:) );
        sumColumns(i) = sum( errorMatrix(:,i) );
    end
    
    for i = 1:5
        PA(i) = errorMatrix(i,i)/sumColumns(i);
        UA(i) = errorMatrix(i,i)/sumRows(i);
    end

    khat = (N*sumCorrect - sum(sumRows.*sumColumns ) ) / (N^2 - sum(sumRows.*sumColumns) );
    
    % Save errors for cross validation
    crossValOA(iteration) = OA;
    crossValPA(:,iteration) = PA;
    crossValUA(:,iteration) = UA;
    crossValk(iteration) = khat;
    crossValErrorMatrix(:,:,iteration) = errorMatrix;
end
% Find average of cross validation errors and save it
averageErrorMatrix = zeros(5,5);
averageOA = sum( crossValOA(:) ) / k;
averagek = sum( crossValk(:)) / k;
for i = 1:5
    averagePA(i) = sum( crossValPA(i,:) ) /k;
    averageUA(i) = sum( crossValUA(i,:) ) /k;
    averageErrorMatrix = averageErrorMatrix + crossValErrorMatrix(:,:,1);
end
averageErrorMatrix = averageErrorMatrix/5;

disp("End of script");
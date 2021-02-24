clc;clear;

% Load data - Split data
data=load('haberman.data');
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);

% Radius
radius = zeros(2,1);
radius = [0.2 0.8];

% Performance metrics for every model
OAMatrix = zeros(4,1);
PAMatrix = zeros(4,2);
UAMatrix = zeros(4,2);
kMatrix  = zeros(4,1);
errorMatrices = zeros(2,2,4);

% r is the index of the radius array
for r = 1:2
    % ANFIS - Scatter Partition - Clustering Per Class
    % Clustering Per Class
    [c1,sig1] = subclust(trnData(trnData(:,end) == 1,:),radius(r));
    [c2,sig2] = subclust(trnData(trnData(:,end) == 2,:),radius(r));
    num_rules = size(c1,1) + size(c2,1);
    
    % Build FIS From Scratch
    fis2 = newfis('FIS_SC','sugeno');
    
    % Add Input-Output Variables
    names_in = {'in1','in2','in3'};
    for i = 1:size(trnData,2)-1
        fis2 = addvar(fis2,'input',names_in{i},[0 1]);
    end
    fis2 = addvar(fis2,'output','out1',[0 1]);
    
    % Add Input Membership Functions
    name = 'sth';
    for i = 1:size(trnData,2)-1
        for j=1:size(c1,1)
            fis2 = addmf(fis2,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]);
        end
        for j=1:size(c2,1)
            fis2 = addmf(fis2,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
        end
    end
    
    % Add Output Membership Functions
    params = [zeros(1,size(c1,1)) ones(1,size(c2,1))];
    for i = 1:num_rules
        fis2 = addmf(fis2,'output',1,name,'constant',params(i));
    end
    
    % Add FIS Rule Base
    ruleList = zeros(num_rules,size(trnData,2));
    for i = 1:size(ruleList,1)
        ruleList(i,:)=i;
    end
    ruleList = [ruleList ones(num_rules,2)];
    fis2 = addrule(fis2,ruleList);
    
    % Train & Evaluate ANFIS
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis2,[100 0 0.01 0.9 1.1],[],chkData);
    figure(2*(r-1)+1);
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    string = 'Class dependent subtractive clustering training error, radius =  ';
    num = num2str(radius(r));
    title1 = strcat(string,num);
    title(title1);
    Y=evalfis(tstData(:,1:end-1),valFis);
    Y=round(Y);
    diff=tstData(:,end)-Y;
    
    % Membership function plots
    for i = 1:size(trnData,2)-1
        num = int2str(i);
        figure(1000*r+i);
        plotmf(valFis,'input',1);
        radiusStr = num2str(radius(r));
        message = 'TSK class dependent r = ';
        message = strcat(message,radiusStr);
        message = strcat(message,', membership functions after training for input ' );
        message = strcat(message,num);
        title(message);   
    end
    
    classes = [1 2];
    errorMatrix = zeros(2,2);
    N = size(tstData,1);
    % Error matrix
    for i = 1:2
        for j = 1:2
            errorMatrix(i,j) = size( intersect( find( Y == classes(i) ) , find(tstData(:,end) == classes(j) ) ) ,1);
        end
    end
    
    % OA
    sumCorrect = trace(errorMatrix);
    OA = 1/N*sumCorrect;
    
    % Producers accuracy and users accuracy
    sumRows = zeros(2,1);
    sumColumns = zeros(2,1);
    PA = zeros(2,1);
    UA = zeros(2,1);
    for i = 1:2
        sumRows(i) = sum( errorMatrix(i,:) );
        sumColumns(i) = sum( errorMatrix(:,i) );
    end
    
    for i = 1:2
        PA(i) = errorMatrix(i,i)/sumColumns(i);
        UA(i) = errorMatrix(i,i)/sumRows(i);
    end

    k = (N*sumCorrect - sum(sumRows.*sumColumns ) ) / (N^2 - sum(sumRows.*sumColumns) );
    
    % Save performaces metrics
    OAMatrix(2*r-1) = OA;
    PAMatrix(2*r-1,:) = PA;
    UAMatrix(2*r-1,:) = UA;
    kMatrix(2*r-1) = k;
    errorMatrices(:,:,2*r-1) = errorMatrix;
    
    % Compare with Class-Independent Scatter Partition
    fis1=genfis2(trnData(:,1:end-1),trnData(:,end),radius(r));
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis1,[100 0 0.01 0.9 1.1],[],chkData);
    figure(2*(r-1)+2);
    plot([trnError valError],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    string = 'Class independent subtractive clustering training error, radius =  ';
    num = num2str(radius(r));
    title2 = strcat(string,num);
    title(title2);
    Y=evalfis(tstData(:,1:end-1),valFis);
    Y=round(Y);
    diff=tstData(:,end)-Y;
    
    % Membership function plots
    for i = 1:size(trnData,2)-1
        num = int2str(i);
        figure(1000*r+100+i);
        plotmf(valFis,'input',1);
        radiusStr = num2str(radius(r));
        message = 'TSK class independent r = ';
        message = strcat(message,radiusStr);
        message = strcat(message,', membership functions after training for input ' );
        message = strcat(message,num);
        title(message);   
    end
    
    classes = [1 2];
    errorMatrix = zeros(2);
    % Error matrix
    for i = 1:2
        for j = 1:2
            errorMatrix(i,j) = size( intersect( find( Y == classes(i) ) , find(tstData(:,end) == classes(j) ) ) ,1);
        end
    end

    % OA
    sumCorrect = trace(errorMatrix);
    OA = 1/size(tstData,1)*sumCorrect;
    
    % Producers accuracy and users accuracy
    sumRows = zeros(2,1);
    sumColumns = zeros(2,1);
    PA = zeros(2,1);
    UA = zeros(2,1);
    for i = 1:2
        sumRows(i) = sum( errorMatrix(i,:) );
        sumColumns(i) = sum( errorMatrix(:,i) );
    end
    
    for i = 1:2
        PA(i) = errorMatrix(i,i)/sumColumns(i);
        UA(i) = errorMatrix(i,i)/sumRows(i);
    end

    k = (N*sumCorrect - sum(sumRows.*sumColumns ) ) / (N^2 - sum(sumRows.*sumColumns) );
    
    % Save performaces metrics
    OAMatrix(2*r) = OA;
    PAMatrix(2*r,:) = PA;
    UAMatrix(2*r,:) = UA;
    kMatrix(2*r) = k;
    errorMatrices(:,:,2*r) = errorMatrix;
    
end
% Display results
disp(OAMatrix);
disp(PAMatrix);
disp(UAMatrix);
disp(kMatrix);
for i = 1:4
    disp(errorMatrices(:,:,i));
end
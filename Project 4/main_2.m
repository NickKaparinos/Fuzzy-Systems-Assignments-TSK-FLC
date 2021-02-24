clc;clear;

disp("Start of script");
% Read data and normalise
data = csvread('data.csv',1,1);
data = normaliseData(data);

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

errors = zeros(a,b);

% k-fold cross validation is used
k = 5;

tic
% Grid search
for w = 1:a
    for z = 1:b
        numOfFeatures = gridSearchParams(w,z,1);
        ra = gridSearchParams(w,z,2);
        
        crossValOA = zeros(k,1);
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
            
            % Train & Evaluate ANFIS
            [trnFis,trnError,~,valFis,valError]=anfis(trnDataFS,fis2,[100 0 0.01 0.9 1.1],[],chkDataFS);
            figure(1000);
            plot([trnError valError],'LineWidth',2); grid on;
            legend('Training Error','Validation Error');
            xlabel('# of Epochs');
            ylabel('Error');
            Y=evalfis(tstDataFS(:,1:end-1),valFis);
            Y=round(Y);
            diff=tstDataFS(:,end)-Y;
            Acc=(length(diff)-nnz(diff))/length(Y)*100;
            
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
            
            % Save error for cross validation
            crossValOA(iteration) = OA
            
            
        end
        % Find average of cross validation errors and save it
        tempErrorOA = sum( crossValOA(:) ) / k;
        errors(w,z) = tempErrorOA; 
    end
end
toc

errors
disp("End of script");
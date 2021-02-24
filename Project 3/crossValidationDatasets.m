function [trnData,chkData,tstData] = crossValidationDatasets(k,data,iteration)
%   crossValidationDatasets
%   this function returns the datasets required for k-fold cross validation
%   data is split 60%/20%/20% into trnData,chkData,tstData respectivly

    n = size(data,1);

    % find tstData and remove them from dataset
    if( iteration ~= k )
        % if it is not the last iteration
        tstData = data( round(n/k)*(iteration-1) + 1 : round(n/k)*iteration , :);
        trnChkDatasets = data;
        trnChkDatasets( round(n/k)*(iteration-1) + 1 : round(n/k)*iteration , : ) = [];
    else
        % if it is the last iteration
        % prevents exceding the length of the array due to rounding
        tstData = data( round(n/k)*(iteration-1) + 1 : end , :);
        trnChkDatasets = data;
        trnChkDatasets( round(n/k)*(iteration-1) + 1 : end , : ) = [];
    end
    
    % find chkData and remove them from dataset
    % what remains are the trnData
    idx = randperm(length(trnChkDatasets));
    chkIdx = idx(1:round(length(idx)*0.25));
    trnIdx = idx(round(length(idx)*0.25)+1:end);
    
    trnData = trnChkDatasets(trnIdx,:);
    chkData = trnChkDatasets(chkIdx,:);
end


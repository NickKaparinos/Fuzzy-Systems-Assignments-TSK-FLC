function normData = normaliseData(data)
    dataTemp = data(:,1:end-1);
    
    xmin = min(dataTemp,[],1);
    xmax = max(dataTemp,[],1);
    dataTemp = (dataTemp-repmat(xmin,[length(dataTemp) 1]))./(repmat(xmax,[length(dataTemp) 1])-repmat(xmin,[length(dataTemp) 1]));
    
    dataTemp = [ dataTemp data(:,end) ];
    normData = dataTemp;
end


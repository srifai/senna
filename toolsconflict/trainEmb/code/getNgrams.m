function allNgrams = getNgrams(allSNum, ksz)
    counter = 0;
    allNgrams = zeros(ksz,10000);
    for i = 1:length(allSNum)
        if mod(i,100000) == 0
            display([num2str(i) ' of ' num2str(length(allSNum))]);
        end
        if length(allSNum{i}) < ksz
            continue
        end

        % setup ngrams
        ngrams = zeros(ksz, length(allSNum{i})-ksz+1);
        for j = 1:ksz
            ngrams(j,:) = allSNum{i}(j:end-ksz + j);
        end

        allNgrams(:,counter+1:counter+size(ngrams,2)) = ngrams;
        counter = counter + size(ngrams,2);
    end
    allNgrams(:,counter+1:end) = [];
end
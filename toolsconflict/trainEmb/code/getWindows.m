function windows = getWindows(ngrams, We)
    dSize = size(We,2);
    
    emb = ceil(ngrams/dSize);
    ngrams = mod(ngrams-1, dSize)+1;
        
    temp = ngrams(:);
    tempe = emb(:)-1;
    windows = We(:,temp+tempe*size(We,2));
    windows = reshape(windows,size(ngrams,1)*size(We,1),size(ngrams,2));
end
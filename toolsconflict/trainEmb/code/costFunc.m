function [cost_total, grad_total] = costFunc(theta, data, params)

[W, b, Wc, bc, We] = stack2param(theta, params.decodeInfo);

Wb2 = W{2};
cWb2 = Wc{2};

Wb1 = [W{1}(:,1:end-params.embeddingSize) b{1}];
Ww = W{1}(:,end-params.embeddingSize+1:end);
W1c = W{1}(:,1:end-params.embeddingSize);

cWb1 = [Wc{1}(:,1:end-params.embeddingSize) bc{1}];
cWw = Wc{1}(:,end-params.embeddingSize+1:end);
cW1c = Wc{1}(:,1:end-params.embeddingSize);

cost_total = 0;

numNgrams = size(data.allngrams,2)/2;

% set up global contexts
docs = min(data.doc):max(data.doc);
tf = data.tf(:,docs);
docI = sparse(data.doc-data.doc(1)+1, 1:numNgrams, 1, length(docs), numNgrams);
docCont = reshape(We,params.embeddingSize,[]) * tf;
a2o = cWb1*[docCont; ones(1,size(docCont,2))];
a2o = a2o*docI;


contexts = data.allngrams(1:end-1,1:end/2);
gwords = data.allngrams(end,1:end/2);
bwords = data.allngrams(end,end/2+1:end);


allWindows = getWindows(contexts, We);

gEmb = getWindows(gwords, We);
bEmb = getWindows(bwords, We);


windows = allWindows;
ngrams = contexts;

% =================================================================
% Feedforward
a1 = [windows; ones(1,size(windows,2))];

% local context
a2a = Wb1*a1;
a2g = params.f(a2a + Ww*gEmb);
a2b = params.f(a2a + Ww*bEmb);
aLg = Wb2*a2g;
aLb = Wb2*a2b;

% global context
ca2g = params.f(a2o + cWw*gEmb);
ca2b = params.f(a2o + cWw*bEmb);
caLg = cWb2*ca2g;
caLb = cWb2*ca2b;

sg = (aLg + caLg)/2;
sb = (aLb + caLb)/2;

scoreDiff = sg - sb;
cost = max(0, 1 - scoreDiff);

% only backprop those with cost > 0
keep = cost > 0;

if cost == 0
    grad_total = zeros(size(theta));
    return
end

windows = windows(:,keep);
a2g = a2g(:,keep);
a2b = a2b(:,keep);
ca2g = ca2g(:,keep);
ca2b = ca2b(:,keep);
gEmb = gEmb(:,keep);
bEmb = bEmb(:,keep);

docI = docI(:,keep);

kl = sum(keep);
ngrams = ngrams(:,keep);
gwords = gwords(:,keep);
bwords = bwords(:,keep);

cost_total = cost_total + sum(cost);

% =================================================================
% Backprop

% local context
temp = -a2g+a2b;
gradW2 = sum(temp,2)';
temp = W{2}(ones(kl,1),:)';
deltag = temp.*params.f_prime(a2g);
deltab = temp.*params.f_prime(a2b);

ddiff = -deltag+deltab;

gradW1c = ddiff*(windows)';        % windows is a1(1:end-1,:)
gradWw = -deltag*(gEmb)' + deltab*(bEmb)';

gradb1 = sum(ddiff,2);

delta = (W1c./2)'*ddiff;
deltag = (Ww./2)'*deltag;
deltab = (Ww./2)'*deltab;

delta0 = reshape(delta, params.embeddingSize, numel(ngrams));
deltag0 = reshape(deltag, params.embeddingSize, numel(gwords));
deltab0 = reshape(deltab, params.embeddingSize, numel(bwords));

allwords = [ngrams(:);gwords(:);bwords(:)];

numTotalWords = numel(allwords);
inputA = sparse(1:numTotalWords,allwords,ones(numTotalWords,1),numTotalWords,size(We,2)*size(We,3));

% global context
temp = -ca2g+ca2b;
gradcW2 = sum(temp,2)';
temp = Wc{2}(ones(kl,1),:)';
deltag = temp.*params.f_prime(ca2g);
deltab = temp.*params.f_prime(ca2b);

ddiff = -deltag+deltab;

gradcWw = -deltag*(gEmb)' + deltab*(bEmb)';

gradcW1c = ddiff*(docCont*docI)';
gradcb1 = sum(ddiff,2);

deltag = (cWw./2)'*deltag;
deltab = (cWw./2)'*deltab;

deltag = reshape(deltag, params.embeddingSize, numel(gwords));
deltab = reshape(deltab, params.embeddingSize, numel(bwords));

temp1 = [delta0, -deltag0-deltag, deltab0+deltab]*inputA;

delta2 = (cW1c./2)'*ddiff;

temp3 = (delta2*docI')*tf';
gradWe = reshape(temp1+temp3, size(We,1),size(We,2),size(We,3));


% =================================================================
gradW1 = [gradW1c gradWw];

gradW2 = 1/numNgrams*gradW2./2;
gradW1 = 1/numNgrams*gradW1./2;
gradb1 = 1/numNgrams*gradb1./2;

gradcW1 = [gradcW1c gradcWw];

gradcW2 = 1/numNgrams*gradcW2./2;
gradcW1 = 1/numNgrams*gradcW1./2;
gradcb1 = 1/numNgrams*gradcb1./2;

gradWe = 1/numNgrams*gradWe;

grad_total = [gradW1(:);gradW2(:);gradb1(:);gradcW1(:);gradcW2(:);gradcb1(:);gradWe(:)];
cost_total = 1/numNgrams*cost_total;


end
clear

addpath(genpath('../toolbox'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vocab file
% Format:
%   Each line should be:
%     <word> <word_id> <word_frequency_in_corpus>
vocabFile = '../data/vocab.txt';
% The following 3 tokens should be in the vocab file. If you use different tokens,
% simple modify the string values of the 3 variables below.
eod = 'eeeoddd';     % this denotes the end-of-document in the corpus files
bpad = '<s>';        % pad for beginning of document
epad = '</s>';       % pad for end of document

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Corpus
% Folder where the corpus files reside and number of files in the folder
% Format:
%   Each line is the word_id that corresponds to a word in the document based on id's in the vocab file.
%   Documents are separated by a line with 'eeeoddd' only (see vocab file section)
% Due to memory limitations, it's best to divide the corpus into multiple smaller files.
corpusFileName = '../data/corpus/';
numFiles = 2;      % number of files in corpusFileName


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Document frequency file
% Format:
%   The first line is the number of documents in the corpus.
%   Then, the n-th line is the number of documents that contain the word with id n-1
dfFile = '../data/df.txt';

% Data file containing the stop words, contains 'stopWords', a cell array of strings
stopWordsFile = '../data/stopWords.mat';

numPasses = 10;     % number of times to go through the corpus for training

saveEvery = 10000;  % save word embeddings every this number of iterations

%% Model Parameters
fields = {{'stopW',             1},      % 1 for removing stop words, 0 for keeping
          {'embeddingSize',     50},     % dimension of the word embeddings
          {'n',                 10},     % number of words before the target word used in local context
          {'layers',            [100]},  % number of units in the hidden layer
          {'batchSize',         1000},   % number of windows to use in each mini-batch during training
};

for i = 1:length(fields)
    if exist('params','var') && isfield(params,fields{i}{1})
        disp(['Warning, we use the previously defined parameter ' fields{i}{1}])
    else
        params.(fields{i}{1}) = fields{i}{2};
    end
end

COST = @costFunc;

params.f = @tanh;
params.f_prime = @tanh_prime;

params.ksz = params.n+1;


%% Load vocab
fid = fopen(vocabFile,'r');
fileLines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
fclose(fid);
fileLines=fileLines{1};

vocab = cell(1,length(fileLines));
freq = zeros(1,length(fileLines));

for i = 1:length(fileLines)
    tempstr = fileLines{i};
    temp=regexp(tempstr,' ','split');
    vocab{str2num(temp{2})} = temp{1};
    freq(str2num(temp{2})) = str2num(temp{3});
end
params.dictionarySize = length(vocab);
display('Finished loading vocab.')

% load stop words
load(stopWordsFile);
sIdx = find(ismember(vocab,stopWords));


%% Initialize parameters
[theta params.decodeInfo] = initializeParameters(params);


%% Train
params.iter = 0;
params.lastSaved = 0;

% minFunc options
options.Method = 'lbfgs';
options.DerivativeCheck = 'off';
options.display = 'off';
options.maxIter = 1;

batchSize = params.batchSize;

% load df file
df = load(dfFile);
numDocs = df(1);
df = df(2:end)';
idf = log(numDocs./(df+1));
dataToUse.idf = idf;

candidates = 1:params.dictionarySize;
candidates = candidates(freq>0);

for loop = 1:numPasses
for filei = 1:numFiles
    display(['File ' num2str(filei)]);
    %======================================================
    % Load corpus
    fid = fopen([corpusFileName num2str(filei) '.txt'], 'r');
    fileLines = textscan(fid, '%s', 'delimiter', '\n');
    fclose(fid);
    fileLines=fileLines{1};
    
        
    splits = [0; find(strcmp(eod,fileLines))];
    allNgrams = zeros(params.ksz, 1000000);
    doc = zeros(1,1000000);
    tf = sparse([],[],[],length(idf), length(splits)-1, length(splits)*2000);
    last = 0;
    
    beginT = repmat({num2str(find(strcmp(bpad,vocab)))},params.n,1);
    endT = repmat({num2str(find(strcmp(epad,vocab)))},params.n,1);

    % setup ngrams and global contexts
    for i = 1:(length(splits)-1)
        allSNum = {cellfun(@str2num,[beginT; fileLines(splits(i)+1:splits(i+1)-1); endT])'};
        
        if params.stopW == 1        % remove stop words
            allSNum{1} = allSNum{1}(~ismember(allSNum{1},sIdx));
        end
        
        allSNum2 = {allSNum{1}(length(beginT)+1:end-length(endT))};
        if isempty(allSNum2{1})
            continue
        end
        
        temp = getNgrams(allSNum, params.ksz);
        allNgrams(:,last+1:last+size(temp,2)) = temp;
        doc(last+1:last+size(temp,2)) = i;
        last = last + size(temp,2);
        
        temp = sparse(ones(1,length(allSNum2{1})),allSNum2{1}(:),idf(allSNum2{1}(:)),1,length(idf));
        temp2 = temp./sum(temp);
        tf(:,i) = temp2;
    end
    allNgrams(:,last+1:end) = [];
    doc(last+1:end) = [];
    
    numNgrams = size(allNgrams,2);
    numBatches = ceil(numNgrams/batchSize);
    display(['Number of batches: ' num2str(numBatches)]);
    
    for batchj = 1:numBatches
        first = 1 + batchSize*(batchj-1);
        last = batchSize + batchSize*(batchj-1);
        if last > numNgrams
            last = numNgrams;
        end
        
        %==================================================================
        % corrupt ngrams
        ngrams = allNgrams(:,first:last);
        cngrams = ngrams;
        cngrams(end,:) = candidates(randi(length(candidates),1,size(ngrams,2)));
            
        dataToUse.allngrams = [ngrams cngrams];
        dataToUse.doc = doc(first:last);
        dataToUse.tf = tf;

        %==================================================================
        % optimize
        [theta, cost, ~, output] = minFunc( @(p) COST(p, dataToUse, params),theta, options);
        
        params.iter = params.iter + output.iterations;
        % intermediate saves
        if (params.iter - params.lastSaved >= saveEvery || ...
                (params.iter <= 10000 && params.iter - params.lastSaved >= 1000))
            params.lastSaved = params.iter;
            
            savedParam = ['iter' num2str(params.iter)];
            save(['../savedParams/' savedParam '.mat'],'theta','params');
        end
    end
end
end









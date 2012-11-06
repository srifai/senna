function [theta decodeInfo] = initializeParameters(params)

% bottom layer
inputSize = params.ksz * params.embeddingSize;

% deep layers
W = cell(length(params.layers)+1,1);
b = cell(length(params.layers),1);           % no bias at last layer

layers = [inputSize params.layers 1];
for i = 1:(length(layers)-1)
    fanIn = layers(i);
    fi = fanIn;
    fi = fi + params.embeddingSize;
    r = 1/sqrt(fi);
    W{i} = rand(layers(i+1), fanIn) * 2*r - r;
    if i ~= length(layers)-1
        b{i} = zeros(layers(i+1), 1);
    end
end

r = 1/sqrt(layers(1) + params.embeddingSize);
Wc{1} = rand(layers(2), params.embeddingSize*2) * 2*r - r;
bc{1} = zeros(layers(2), 1);

fanIn = layers(2);
r = 1/sqrt(fanIn);
Wc{2} = rand(layers(3), fanIn) * 2*r - r;


% word embeddings
r  = 0.01;
We = rand(params.embeddingSize, params.dictionarySize) * 2 * r - r;

[theta decodeInfo] = param2stack(W,b,Wc,bc,We);

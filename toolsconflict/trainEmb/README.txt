##########################################################################################
# Word Vectors Training Code
# 
# Improving Word Representations via Global Context and Multiple Word Prototypes
# Annual Meeting of the Association for Computational Linguistics (ACL), 2012.
# Eric H. Huang, Richard Socher, Christopher D. Manning, Andrew Y. Ng
# Computer Science Department
# Stanford University
##########################################################################################


This package is available for download at
http://ai.stanford.edu/~ehhuang/. It includes the training code for
the global-context-aware neural language model described in the paper.

Please cite the paper if you use the code:

@inproceedings{HuangEtAl2012,
author = {Eric H. Huang and Richard Socher and Christopher D. Manning and Andrew Y. Ng},
title = {Improving Word Representations via Global Context and Multiple Word Prototypes},
booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL)},
year = 2012
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                   %
%                Running the Code                   %
%                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

- You can train the word embeddings by running

  ./code/trainEmb.m

  in Matlab.

- To train word embeddings on your own corpus, you need to provide 3 files, as described in the comments
  at the top of ./code/trainEmb.m

- Examples of these 3 files are also included in this package.

- After training, you can get the word embeddings by running the following commands in Matlab.

  load('savedParams/iter<N>.mat');
  [~, ~, ~, ~, We] = stack2param(theta, params.decodeInfo);

  You need to replace <N> with the actual iteration number you want to load. We is the matrix
  of the word embeddings where column i is the embedding for word i.





- Note that the code is highly optimized for speed, so it might be hard to understand.

---------------------------------------------------------------------------
last modified: 7/27/2012

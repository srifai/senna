import numpy
import theano
import theano.tensor as T
from util.sparse import idx2spmat
import pdb
def error(sentences, scorefct, embeddings, wsize):
    nsent = len(sentences)
    delta = wsize/2
    rest = wsize%2
    rank = []
    for i,s in enumerate(sentences):
        nword = len(s)
        if nword < wsize + 2:
            continue
        for j in range(delta,nword-delta):
            chunks = []
            st = s[j-delta:j]
            en = s[j+1:j+delta+1]
            for k in range(embeddings):
                chunks +=  st + [k] + en

            score = scorefct(idx2spmat(chunks,embeddings))
            sortedscore = numpy.argsort(score[::-1].flatten())
            rank += [ numpy.argwhere(sortedscore==j).flatten()[0] ]

    return numpy.mean(rank)
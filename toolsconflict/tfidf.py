import numpy
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.io import loadmat
import pdb

def count(doc, N=30000):
    c = numpy.zeros((N,))
    for w in doc:
        c[w]+=1
    return c

def idf(corpus, N=30000):
    occ = numpy.sign(count(corpus[0],N))
    for d in corpus[1:]:
        occ += numpy.sign(count(d,N))
    return numpy.log(len(corpus) * (1./ (occ+1)) )

def tf(doc, N=30000):
    c = count(doc,N)
    return c/numpy.max(c)
        

if __name__ == "__main__":
    corpus = cPickle.load(open('/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004_prepro30k.pkl'))
    pdb.set_trace()
    idfvect = idf(corpus)
    numpy.save('wiki_idf_vect.npy',idfvect)

import numpy
import cPickle
import scipy
import csv
import scipy.stats
import pdb
import os
import theano
import theano.tensor as T
import sys
from dataset.DatasetInterfaces import root

from util.embedding import knn, display
from util.cost import nll, hardmax, cepond, nllsoft,ce
from util.expr import rect, identity, hardtanh
from util.io import save, load
from util.sparse import idx2spmat, idx2mat, idx2vec

from unsupervised import cae,ae
from supervised import logistic
from jobscheduler import JobmanInterface as JB
from tools.hyperparams import hyperparams

import sparse.supervised

from theano import function

def parse_data(f='combined.csv', vocab='/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004_vocab30k.pkl'):
    vocab = cPickle.load(open(vocab))
    print len(vocab)
    dvocab = dict(zip(vocab,range(len(vocab))))
    def map2vocab(s):
        cidx = []
        for w in s:
            try:
                cidx.append(dvocab[w])
            except:
                #print 'Word not found:',w
                cidx.append(dvocab['UUUKKKNNN'])
        return cidx


    comb = open(f,'r')
    combr = csv.reader(comb)

    wordsim = []
    combr.next()
    for row in combr:
        # ['tiger', 'cat', '7.35']
        w1,w2,s = row
        s = float(s)
        w1idx = map2vocab([w1])[0]
        w2idx = map2vocab([w2])[0]
        element = (w1idx,w2idx,s)
        wordsim.append(element)
    return wordsim

def score(jobman,path):
    hp = jobman.state
    nsenna = 30000

    PATH = '/scratch/rifaisal/msrtest/test/'

    embedding = cae(i_size=nsenna, h_size=hp['embedsize'], e_act = identity)
    load(embedding,path+'/embedding.pkl')

    words = parse_data()
    scores = []
    esims = []
    msim = []
    hsim = []
    Em = embedding.params['e_weights'].get_value(borrow=True)
    for i,(w1,w2,s) in enumerate(words):
        sys.stdout.flush()

        w1em = Em[w1]
        w2em = Em[w2]

        esim = -((w1em - w2em)**2).sum()
        esims.append(esim)
        hsim.append(s)
                           
    print 'Embeddings:',scipy.stats.spearmanr(numpy.array(hsim), numpy.array(esims))[0]
    
def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    for i in range(1,13):
        path = state['loadpath']+'/'+str(i)+'/files'
        print '---------Computing:',i,
        score(jobhandler,path)
    return 


if __name__ == "__main__":
    HP_init = [ ('values','epoch',[100]),
                ('values','deviation',[.1]),
                ('values','iresume',[3756]),
                ('values','freq',[10000]),
                ('values','loadpath',['/scratch/rifaisal/exp/mullerx_db/wikibaseline_bugfix_0001_resume_5/']),
                ('values','hsize',[100]),
                ('values','embedsize',[50]),
                ('values','wsize',[9]),
                ('values','npos',[1]),
                ('values','nneg',[1,10]),
                ('values','lr',[0.01,.1]),
                ('values','lambda',[.0]),
                ('values','margin',[1.]),
                ('values','bs',[10]) ]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)


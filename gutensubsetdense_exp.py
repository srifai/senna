from dataset.DatasetInterfaces import root

from util.embedding import knn, display
from util.cost import nll, hardmax
from util.expr import rect, identity, hardtanh
from util.io import save, load
from util.sparse import idx2spmat, idx2mat

from unsupervised import cae,ae
from supervised import logistic
from jobscheduler import JobmanInterface as JB
from tools.hyperparams import hyperparams

import sparse.supervised
import evaluation

import theano
import theano.sparse
import theano.tensor as T
from theano import function

import numpy
import sys,pdb,cPickle
import time

def run(jobman,debug = False):
    hp = jobman.state

    # Symbolic variables

    s_posit = T.matrix()#theano.sparse.csr_matrix()
    s_negat = T.matrix()#theano.sparse.csr_matrix()

    s_valid = theano.sparse.csr_matrix()

    sentences = cPickle.load(open('/data/lisatmp2/rifaisal/guten_subset_idx.pkl'))

    validsentence = sentences[-10:]
    sentences = sentences[:-10]
    senna = cPickle.load(open('/data/lisatmp2/rifaisal/senna.pkl'))
    gsubset = cPickle.load(open('/data/lisatmp2/rifaisal/guten_vocab_subset.pkl')).flatten().tolist()
    hashtab = dict( zip( gsubset, range( len( gsubset))))    

    senna = numpy.array(senna)[gsubset].tolist()

    nsent = len(sentences)
    nsenna = len(senna)

    # Layers
    embedding = logistic(i_size=nsenna, h_size=hp['embedsize'], act = identity)
    H = ae(i_size = hp['embedsize']*hp['wsize'], h_size=hp['hsize'], e_act = rect, d_act = hardtanh)
    L = logistic(i_size = hp['hsize'],  h_size = 1)#, act = identity)

    valid_embedding = sparse.supervised.logistic(i_size=nsenna, h_size=hp['embedsize'], act = identity)
    #valid_embedding.params['weights'].set_value(embedding.params['weights'].get_value(borrow=True))
    #valid_embedding.params['bias'].set_value(embedding.params['bias'].get_value(borrow=True))


    lr = hp['lr']
    h_size = hp['hsize']
    bs = hp['bs']

    posit_embed = embedding.encode(s_posit).reshape((1,hp['embedsize']*hp['wsize']))
    negat_embed = embedding.encode(s_negat).reshape((hp['nneg'],hp['embedsize']*hp['wsize']))
    #valid_embed = valid_embedding.encode(s_valid).reshape((nsenna,hp['embedsize']*hp['wsize']))


    posit_score = L.encode(H.encode(posit_embed))
    negat_score = L.encode(H.encode(negat_embed))
    #valid_score = L.encode(H.encode(valid_embed))

    C = (negat_score - posit_score.flatten() + hp['margin'])

    CC = (rect(C)).mean()

    opt = theano.function([s_posit, s_negat], 
                          C.mean(), 
                          updates = dict( L.update(CC,lr) + H.update(CC,lr) + embedding.update_norm(CC,lr)) )

    #validfct = theano.function([s_valid],valid_score)

    #print 'Random Valid Mean rank',evaluation.error(validsentence, validfct, nsenna, hp['wsize'])

    #load(valid_embedding,'files/gutensubsetdense_exp.py_embedding.pkl')
    load(embedding,'files/gutensubsetdense_exp.py_embedding.pkl')
    load(H,'files/gutensubsetdense_exp.py_hidden.pkl')
    load(L,'files/gutensubsetdense_exp.py_logistic.pkl')

    delta = hp['wsize']/2
    rest = hp['wsize']%2

    freq_idx = cPickle.load(open('/data/lisatmp2/rifaisal/gutten_sorted_vocab.pkl'))[:1000]
    freq_idx =  [ hashtab[idx] for idx in freq_idx ]

    fname = sys.argv[0]+'_'
    

    for e in range(hp['epoch']):
        c = []
        for i in range(nsent):
            rsent = numpy.random.randint(nsent-1)
            nword = len(sentences[rsent])
            if nword < hp['wsize'] + 2:
                continue

            pidx = numpy.random.randint(low = delta, high = nword-delta)
            pchunk = sentences[rsent][pidx-delta:pidx+delta+rest]
            nchunk = []
            st = sentences[rsent][pidx-delta:pidx]
            en = sentences[rsent][pidx+1:pidx+delta+rest]
            rndidx = numpy.random.randint(nsenna, size = (hp['nneg'],))
            nchunk = []
            for j in range(hp['nneg']):
                nchunk += en + [rndidx[j]] + st


            assert len(nchunk) == len(pchunk)*hp['nneg']
            #start = time.time()
            p, n = (idx2mat(pchunk,nsenna), idx2mat(nchunk,nsenna))
            #print 'Select row:',time.time()-start,
            #start = time.time()
            c.append(opt(p,n))
            #print 'grad up:',time.time()-start

            if i%hp['freq'] == 0:
                print e,i, numpy.array(c).mean(0)
                ne = knn(freq_idx,embedding.params['weights'].get_value(borrow=True))
                save(embedding,fname+'embedding.pkl')
                save(H,fname+'hidden.pkl')
                save(L,fname+'logistic.pkl')
                sys.stdout.flush()
                open('files/'+fname+'nearest.txt','w').write(display(ne,senna))

    #print 'Valid Mean rank',evaluation.error(validsentence, validfct, nsenna, hp['wsize'])
    save(embedding,fname+'embedding.pkl')
    save(H,fname+'hidden.pkl')
    save(L,fname+'logistic.pkl')

def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    run(jobhandler,False)
    return 0


if __name__ == "__main__":
    HP_init = [ ('values','dataset',['/data/lisatmp/mullerx/jfk3/amz_dvd_3_files_dict.pkl']),
                ('values','adjmat',['/data/lisatmp/mesnilgr/datasets/jfkd/adjacency/adjacenty_lil_matrix_indomain_movies.npy']),
                ('values','epoch',[2]),
                ('values','freq',[10000]),
                ('values','hsize',[300]),
                ('values','embedsize',[50]),
                ('values','wsize',[7]),
                ('values','npos',[1]),
                ('values','nneg',[100]),
                ('values','lr',[0.0001]),
                ('values','lambda',[.1]),
                ('values','margin',[1.]),
                ('values','bs',[10]) ]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)




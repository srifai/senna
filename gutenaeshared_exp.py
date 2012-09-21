from dataset.DatasetInterfaces import root

from util.embedding import knn, display
from util.cost import nll, hardmax
from util.expr import rect, identity, hardtanh
from util.io import save, load
from util.sparse import idx2spmat, idx2mat, idx2vec

from unsupervised import cae,ae
from supervised import logistic
from jobscheduler import JobmanInterface as JB
from tools.hyperparams import hyperparams

import sampler
import sparse.supervised

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
    s_bow = T.matrix()
    s_posit = T.matrix()#theano.sparse.csr_matrix()
    s_negat = T.matrix()#theano.sparse.csr_matrix()

    sentences = cPickle.load(open('/data/lisatmp2/rifaisal/guten_subset_idx.pkl'))

    senna = cPickle.load(open('/data/lisatmp2/rifaisal/senna.pkl'))
    gsubset = cPickle.load(open('/data/lisatmp2/rifaisal/guten_vocab_subset.pkl')).flatten().tolist()
    hashtab = dict( zip( gsubset, range( len( gsubset))))    

    senna = numpy.array(senna)[gsubset].tolist()

    nsent = len(sentences)
    nsenna = len(senna)

    # Layers
    embedding = cae(i_size=nsenna, h_size=hp['embedsize'], e_act = T.nnet.sigmoid)
    H = ae(i_size = hp['embedsize']*hp['wsize'], h_size=hp['hsize'], e_act = rect, d_act = hardtanh)
    L = logistic(i_size = hp['hsize'],  h_size = 1)


    lr = hp['lr']
    h_size = hp['hsize']
    bs = hp['bs']

    posit_embed = embedding.encode(s_posit).reshape((1,hp['embedsize']*hp['wsize']))
    negat_embed = embedding.encode(s_negat).reshape((hp['nneg'],hp['embedsize']*hp['wsize']))

    posit_score = L.encode(H.encode(posit_embed))
    negat_score = L.encode(H.encode(negat_embed))

    C = (negat_score - posit_score.flatten() + hp['margin'])

    rec = embedding.reconstruct(s_bow, loss='ce')
    CC = (rect(C)).mean() + hp['lambda'] * rec

    opt = theano.function([s_posit, s_negat, s_bow], 
                          [C.mean(),rec], 
                          updates = dict( L.update(CC,lr) + H.update(CC,lr) + embedding.update(CC,lr)) )



    delta = hp['wsize']/2
    rest = hp['wsize']%2

    freq_idx = cPickle.load(open('/data/lisatmp2/rifaisal/gutten_sorted_vocab.pkl'))[:1000]
    freq_idx =  [ hashtab[idx] for idx in freq_idx ]

    fname = sys.argv[0]+'_'
    

    for e in range(hp['epoch']):
        c = []
        r = []
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
            p, n, b = (idx2mat(pchunk,nsenna), idx2mat(nchunk,nsenna), idx2vec(sentences[rsent],nsenna))
            #print 'Select row:',time.time()-start,
            #start = time.time()
            l,g = opt(p,n,b)
            c.append(l)
            r.append(g)
            #print 'grad up:',time.time()-start

            if i%hp['freq'] == 0:
                print e,i,'local:', numpy.array(c).mean(0), 'global:',numpy.array(r).mean(0)
                ne = knn(freq_idx,embedding.params['e_weights'].get_value(borrow=True))
                save(embedding,fname+'embedding.pkl')
                save(H,fname+'hidden.pkl')
                save(L,fname+'logistic.pkl')
                sys.stdout.flush()
                open('files/'+fname+'nearest.txt','w').write(display(ne,senna))

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
                ('values','epoch',[100]),
                ('values','freq',[1000]),
                ('values','hsize',[100]),
                ('values','embedsize',[60]),
                ('values','wsize',[5]),
                ('values','npos',[1]),
                ('values','nneg',[10]),
                ('values','lr',[0.001]),
                ('values','lambda',[.1]),
                ('values','margin',[1.]),
                ('values','bs',[10]) ]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)




from dataset.DatasetInterfaces import root

from util.embedding import knn, display
from util.cost import nll, hardmax
from util.expr import rect, identity, hardtanh
from util.io import save, load
from util.sparse import idx2spmat, idx2mat, idx2vec

import evaluation
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
    expstart = time.time()
    hp = jobman.state

    # Symbolic variables
    s_bow = T.matrix()
    s_posit = T.matrix()#theano.sparse.csr_matrix()
    s_negat = T.matrix()#theano.sparse.csr_matrix()

    sentences = cPickle.load(open('/scratch/rifaisal/data/guten/guten_subset_idx.pkl'))

    senna = cPickle.load(open('/scratch/rifaisal/data/guten/senna.pkl'))
    gsubset = cPickle.load(open('/scratch/rifaisal/data/guten/guten_vocab_subset.pkl')).flatten().tolist()
    hashtab = dict( zip( gsubset, range( len( gsubset))))    

    senna = numpy.array(senna)[gsubset].tolist()
    s_valid = theano.sparse.csr_matrix()

    validsentence = sentences[-10:]
    sentences = sentences[:-10]




    nsent = len(sentences)
    nsenna = len(senna)

    # Layers
    
    embedding = cae(i_size=nsenna, h_size=hp['embedsize'], e_act = T.nnet.sigmoid)
    H = ae(i_size = hp['embedsize']*hp['wsize'], h_size=hp['hsize'], e_act = rect, d_act = hardtanh)
    L = logistic(i_size = hp['hsize'],  h_size = 1)

    valid_embedding = sparse.supervised.logistic(i_size=nsenna, h_size=hp['embedsize'], act = T.nnet.sigmoid)
    valid_embedding.params['weights'] = embedding.params['e_weights']
    valid_embedding.params['bias'] = embedding.params['e_bias']

    lr = hp['lr']
    h_size = hp['hsize']
    bs = hp['bs']

    posit_embed = embedding.encode(s_posit).reshape((1,hp['embedsize']*hp['wsize']))
    negat_embed = embedding.encode(s_negat).reshape((hp['nneg'],hp['embedsize']*hp['wsize']))
    valid_embed = valid_embedding.encode(s_valid).reshape((nsenna,hp['embedsize']*hp['wsize']))

    posit_score = L.encode(H.encode(posit_embed))
    negat_score = L.encode(H.encode(negat_embed))
    valid_score = L.encode(H.encode(valid_embed))

    C = (negat_score - posit_score.flatten() + hp['margin'])

    rec = embedding.reconstruct(s_bow, loss='ce')
    CC = (rect(C)).mean() + hp['lambda'] * rec

    opt = theano.function([s_posit, s_negat, s_bow], 
                          [C.mean(),rec], 
                          updates = dict( L.update(CC,lr) + H.update(CC,lr) + embedding.update(CC,lr)) )

    validfct = theano.function([s_valid],valid_score)

    def saveexp():
        save(embedding,fname+'embedding.pkl')
        save(H,fname+'hidden.pkl')
        save(L,fname+'logistic.pkl')
        print 'Saved successfully'

    delta = hp['wsize']/2
    rest = hp['wsize']%2

    freq_idx = cPickle.load(open('/scratch/rifaisal/data/guten/gutten_sorted_vocab.pkl'))[:1000]
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

            p, n, b = (idx2mat(pchunk,nsenna), idx2mat(nchunk,nsenna), idx2vec(sentences[rsent],nsenna))

            l,g = opt(p,n,b)
            c.append(l)
            r.append(g)
            
            if (time.time() - expstart) > ( 3600 * 24 * 6 + 3600*20) or (i+1)%(50*hp['freq']) == 0:
                mrk = evaluation.error(validsentence, validfct, nsenna, hp['wsize'])
                hp['mrk'] = mrk
                hp['e'] = e
                hp['i'] = i
                jobman.save()
                saveexp()
                print 'Random Valid Mean rank',mrk

            if i%hp['freq'] == 0:
                hp['score'] = numpy.array(c).mean()
                hp['rec'] = numpy.array(r).mean()
                print e,i,'NN Score:', hp['score'], 'Reconstruction:', hp['rec']

                ne = knn(freq_idx,embedding.params['e_weights'].get_value(borrow=True))
                open('files/'+fname+'nearest.txt','w').write(display(ne,senna))

                saveexp()
                sys.stdout.flush()
                jobman.save()
                
    save()

def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    run(jobhandler,False)
    return 0


if __name__ == "__main__":
    HP_init = [ ('values','dataset',['/data/lisatmp/mullerx/jfk3/amz_dvd_3_files_dict.pkl']),
                ('values','adjmat',['/data/lisatmp/mesnilgr/datasets/jfkd/adjacency/adjacenty_lil_matrix_indomain_movies.npy']),
                ('values','epoch',[100]),
                ('values','freq',[10000]),
                ('values','hsize',[100]),
                ('values','embedsize',[50,100,200]),
                ('values','wsize',[5]),
                ('values','npos',[1]),
                ('values','nneg',[10]),
                ('values','lr',[0.001,.0001,.01]),
                ('values','lambda',[.01,.001,.0001]),
                ('values','margin',[1.]),
                ('values','bs',[10]) ]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)




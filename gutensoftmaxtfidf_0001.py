from dataset.DatasetInterfaces import root

from util.embedding import knn, display
from util.cost import nll, hardmax, cepond, nllsoft,ce
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
import theano.sparse as sp
import theano.tensor as T
from theano import function

import scipy.sparse
import numpy
import sys,pdb,cPickle,os
import time

def run(jobman,debug = False):
    expstart = time.time()
    hp = jobman.state

    if not os.path.exists('files/'): os.mkdir('files/')

    # Symbolic variables
    s_bow = T.matrix()
    s_idx = T.iscalar()
    s_tf = T.scalar()
    s_posit = T.matrix()#theano.sparse.csr_matrix()
    s_negat = T.matrix()#theano.sparse.csr_matrix()

    sentences = cPickle.load(open('/scratch/rifaisal/data/guten/guten_subset_idx.pkl'))

    senna = cPickle.load(open('/scratch/rifaisal/data/guten/senna.pkl'))
    gsubset = cPickle.load(open('/scratch/rifaisal/data/guten/guten_vocab_subset.pkl')).flatten().tolist()
    hashtab = dict( zip( gsubset, range( len( gsubset))))    

    tfidf_data = numpy.load('/scratch/rifaisal/data/guten/guten_tfidf.npy').item().tocsr().astype('float32')

    #tfidf = cPickle.load(open('/scratch/rifaisal/repos/senna/gutentokenizer.pkl'))

    senna = numpy.array(senna)[gsubset].tolist()
    s_valid = theano.sparse.csr_matrix()

    validsentence = sentences[10000:10010]


    nsent = len(sentences)
    nsenna = len(senna)

    # Layers
    
    embedding = cae(i_size=nsenna, h_size=hp['embedsize'], e_act = identity)

    H = ae(i_size = hp['embedsize']*hp['wsize'], h_size=hp['hsize'], e_act = T.tanh)
    L = logistic(i_size = hp['hsize'], h_size = 1, act = identity)
    S = logistic(i_size = hp['embedsize'], h_size = nsenna, act= T.nnet.softmax)


    valid_embedding = sparse.supervised.logistic(i_size=nsenna, h_size=hp['embedsize'], act = identity)
    valid_embedding.params['weights'] = sp.shared(value = scipy.sparse.csr_matrix(embedding.params['e_weights'].get_value(borrow=True)))
    valid_embedding.params['bias'] = embedding.params['e_bias']

    lr = hp['lr']
    h_size = hp['hsize']
    bs = hp['bs']

    posit_embed = T.dot(s_posit, embedding.params['e_weights']).reshape((1,hp['embedsize']*hp['wsize']))
    negat_embed = T.dot(s_negat, embedding.params['e_weights']).reshape((hp['nneg'],hp['embedsize']*hp['wsize']))
    valid_embed = sp.dot(s_valid,valid_embedding.params['weights']).reshape((nsenna,hp['embedsize']*hp['wsize']))

    posit_score = L.encode(H.encode(posit_embed))
    negat_score = L.encode(H.encode(negat_embed))
    valid_score = L.encode(H.encode(valid_embed))

    C = (negat_score - posit_score.flatten() + hp['margin'])

    s_bow_pred = S.encode(embedding.encode(s_bow))


    pred = s_tf * nllsoft(s_bow_pred,s_idx)
    
    CC = (rect(C)).mean() + hp['lambda'] * pred

    opt = theano.function([s_posit, s_negat, s_bow, s_idx, s_tf], 
                          [(rect(C)).mean(),pred], 
                          updates = dict( S.update(CC,lr) + L.update(CC,lr) + H.update(CC,lr) + embedding.update_norm(CC,lr)) )

    #validfct = theano.function([s_valid],valid_score)

    def saveexp():
        save(embedding,fname+'embedding.pkl')
        save(H,fname+'hidden.pkl')
        save(L,fname+'logistic.pkl')

    delta = hp['wsize']/2
    rest = hp['wsize']%2

    freq_idx = cPickle.load(open('/scratch/rifaisal/data/guten/gutten_sorted_vocab.pkl'))[:1000]
    freq_idx =  [ hashtab[idx] for idx in freq_idx ]

    fname = ''
    
    for e in range(hp['epoch']):
        c = []
        r = []
        count = 1
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
            tfidf_chunk = tfidf_data[rsent:rsent+1].toarray()
            #pdb.set_trace()
            tfidf_value = tfidf_chunk[0,sentences[rsent][pidx]]
            tfidf_chunk[0,sentences[rsent][pidx]] = 0.
            tfidx = sentences[rsent][pidx] # numpy.zeros(tfidf_chunk.shape).astype('float32')
            #tfidx[0,sentences[rsent][pidx]] = 1.
            p, n, b, iidx, tfval = (idx2mat(pchunk,nsenna), idx2mat(nchunk,nsenna), tfidf_chunk, tfidx, tfidf_value )
            count += tfval!=0
            l,g = opt(p,n,b, iidx, tfval)
            c = c
            c.append(l)
            r.append(g)

            """
            if (time.time() - expstart) > ( 3600 * 24 * 6 + 3600*20) or (i+1)%(20*hp['freq']) == 0 and debug==False:
                valid_embedding.params['weights'] = sp.shared(value = scipy.sparse.csr_matrix(embedding.params['e_weights'].get_value(borrow=True)))
                mrk = evaluation.error(validsentence, validfct, nsenna, hp['wsize'])
                hp['mrk'] = mrk
                jobman.save()
                saveexp()
                print 'Random Valid Mean rank',mrk
            """

            if (i+1)%hp['freq'] == 0 or debug:
                hp['score'] = numpy.array(c).sum() / (numpy.array(c)>0).sum()
                hp['pred'] = numpy.array(r).sum()/float(count)
                hp['e'] = e
                hp['i'] = i
                print ''
                print e,i,'NN Score:', hp['score'], 'Reconstruction:', hp['pred']

                if debug != True:
                    ne = knn(freq_idx,embedding.params['e_weights'].get_value(borrow=True))
                    open('files/'+fname+'nearest.txt','w').write(display(ne,senna))
                    saveexp()
                sys.stdout.flush()
                jobman.save()
                
    saveexp()

def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    run(jobhandler,False)
    return 0


if __name__ == "__main__":
    HP_init = [ ('values','epoch',[100]),
                ('values','deviation',[.1]),
                ('values','freq',[10000]),
                ('values','hsize',[100]),
                ('values','embedsize',[50,100,200]),
                ('values','wsize',[5]),
                ('values','npos',[1]),
                ('values','nneg',[10]),
                ('values','lr',[0.01,.0001,.001]),
                ('values','lambda',[.1,.01]),
                ('values','margin',[1.]),
                ('values','bs',[10]) ]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)




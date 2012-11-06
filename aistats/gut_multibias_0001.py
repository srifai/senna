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
import sparse.supervised

import theano
import theano.sparse as sp
import theano.tensor as T
from theano import function
from theano.ifelse import ifelse

import scipy.sparse
import numpy
import sys,pdb,cPickle,os,copy
import time

from toolsconflict.tfidf import tf

def run(jobman,debug = False):
    expstart = time.time()
    hp = jobman.state

    if not os.path.exists('files/'): os.mkdir('files/')

    # Symbolic variables
    s_posit = T.matrix()
    s_negat = T.matrix()
    idx_start = T.lscalar()
    idx_stop = T.lscalar()
    s_valid = theano.sparse.csr_matrix()



    w2i = cPickle.load(open('/scratch/rifaisal/data/gutenberg_aistats/merged_word2idx.pkl'))
    i2w = dict( (v,k) for k,v in w2i.iteritems() )
    i2w[0] = 'UNK'
    senna = [ i2w[i] for i in range(len(i2w.keys())) ]


    nsenna = len(senna)
    
    embedding = cae(i_size=nsenna, h_size=hp['embedsize'], e_act = identity)
    H = ae(i_size = hp['embedsize']*hp['wsize'], h_size=hp['hsize'], e_act = T.tanh)
    L = logistic(i_size = hp['hsize'], h_size = 1, act = identity)
 
    minsize = hp['minsize']
    maxsize = hp['maxsize']

    dsize = maxsize - minsize +1

    H.params['e_bias'] = theano.shared( numpy.array(numpy.zeros((dsize,hp['hsize'])),dtype=theano.config.floatX),name='e_bias')

    path = hp['loadpath']
 
    if path:
        load(embedding,path+'/embedding.pkl')
        #load(H,path+'/hidden.pkl')
        #load(L,path+'/logistic.pkl')
        hp['embedsize'] = embedding.params['e_weights'].get_value(borrow=True).shape[1]
        #hp['hsize'] = H.params['e_weights'].get_value(borrow=True).shape[1]
        jobman.save()

    H.params['e_bias'] = theano.shared( numpy.array(numpy.zeros((dsize,hp['hsize'])),dtype=theano.config.floatX),name='e_bias')
    valid_embedding = sparse.supervised.logistic(i_size=nsenna, h_size=hp['embedsize'], act = identity)
    valid_embedding.params['weights'] = sp.shared(value = scipy.sparse.csr_matrix(embedding.params['e_weights'].get_value(borrow=True)))
    valid_embedding.params['bias'] = embedding.params['e_bias']

    lr = hp['lr']
    h_size = hp['hsize']
    bs = hp['bs']

    posit_embed = T.dot(s_posit, embedding.params['e_weights']).reshape((1,hp['embedsize']*hp['wsize']))
    negat_embed = T.dot(s_negat, embedding.params['e_weights']).reshape((hp['nneg'],hp['embedsize']*hp['wsize']))
    valid_embed = sp.dot(s_valid,valid_embedding.params['weights']).reshape((nsenna,hp['embedsize']*hp['wsize']))

    posit_embed_left = T.concatenate([posit_embed[:,idx_start*hp['embedsize']:idx_stop*hp['embedsize']],
                                  T.zeros_like(posit_embed[:,idx_stop*hp['embedsize']:]) ],axis=1)

    negat_embed_left = T.concatenate([negat_embed[:,idx_start*hp['embedsize']:idx_stop*hp['embedsize']],
                                   T.zeros_like(negat_embed[:,idx_stop*hp['embedsize']:]) ],axis=1)

    posit_embed_right = T.concatenate([ T.zeros_like(posit_embed[:,:idx_start*hp['embedsize']]),
                                  posit_embed[:,idx_start*hp['embedsize']:idx_stop*hp['embedsize']]],axis=1)

    negat_embed_right = T.concatenate([ T.zeros_like(negat_embed[:,:idx_start*hp['embedsize']]),
                                   negat_embed[:,idx_start*hp['embedsize']:idx_stop*hp['embedsize']]],axis=1)



    posit_embed = T.concatenate([ T.zeros_like(posit_embed[:,:idx_start*hp['embedsize']]),
                                  posit_embed[:,idx_start*hp['embedsize']:idx_stop*hp['embedsize']],
                                  T.zeros_like(posit_embed[:,idx_stop*hp['embedsize']:]) ],axis=1)

    negat_embed = T.concatenate([ T.zeros_like(negat_embed[:,:idx_start*hp['embedsize']]),
                                   negat_embed[:,idx_start*hp['embedsize']:idx_stop*hp['embedsize']],
                                   T.zeros_like(negat_embed[:,idx_stop*hp['embedsize']:]) ],axis=1)

    
    #posit_embed = ifelse(T.eq(idx_start, 0), posit_embed_left, posit_embed)
    #posit_embed = ifelse(T.eq(idx_stop, hp['maxsize']), posit_embed_right, posit_embed)

    #negat_embed = ifelse(T.eq(idx_start, 0), negat_embed_left, negat_embed)
    #negat_embed = ifelse(T.eq(idx_stop, hp['maxsize']), negat_embed_right, negat_embed)

    Hposit = T.tanh(T.dot(posit_embed,H.params['e_weights']) + H.params['e_bias'][idx_stop-idx_start-minsize,:])
    Hnegat = T.tanh(T.dot(negat_embed,H.params['e_weights']) + H.params['e_bias'][idx_stop-idx_start-minsize,:])
    posit_score = L.encode(Hposit)
    negat_score = L.encode(Hnegat)
    valid_score = L.encode(H.encode(valid_embed))

    C = (negat_score - posit_score.flatten() + hp['margin'])

    CC = (rect(C)).mean()

    opt = theano.function([s_posit, s_negat, idx_start, idx_stop],
                          (rect(C)).mean(),
                          updates = dict( L.update(CC,lr) + H.update(CC,lr) + embedding.update_norm(CC,lr)) )

    validfct = theano.function([s_valid],valid_score)

    def saveexp():
        save(embedding,fname+'embedding.pkl')
        save(H,fname+'hidden.pkl')
        save(L,fname+'logistic.pkl')

    delta = hp['wsize']/2
    rest = hp['wsize']%2

    freq_idx = cPickle.load(open('/scratch/rifaisal/data/gutenberg_aistats/sorted_vocab.pkl'))[:2000]
    fname = ''
    validsentence = cPickle.load(open('/scratch/rifaisal/data/wiki_april_2010/valid_debug.pkl'))
    tseenwords = not debug
    for e in range(hp['epoch']):
        hp['split'] = numpy.random.randint(45)
        sentences = cPickle.load(open('/scratch/rifaisal/data/gutenberg_aistats/split'+str(hp['split'])+'.pkl'))
        nsent = len(sentences)
        bigc = []
        bigr = []

        seen_words = 0
        for i,s in enumerate(sentences):
            nword = len(s)
            seen_words += nword
            tseenwords += nword

            if nword < hp['maxsize'] + 2:
                continue
            rndsize = numpy.random.randint(low=hp['minsize']+1,high=hp['maxsize']-1)
            idxsta = numpy.random.randint(low=1, high=hp['maxsize']-rndsize)
            idxsto = idxsta+rndsize

            print 'r',rndsize,'b',idxsta,'e',idxsto,'shape',H.params['e_bias'].get_value().shape

            c =[]
            r =[]
            if debug:
                print ' *** Processing document',i,'with',nword,
                sys.stdout.flush()
            for j in range(delta,nword-delta):
                nd = rndsize/2
                rd = rndsize%2
                pchunk = s[j-delta:j+delta+rest]
                nchunk = []
                
                rndidx = numpy.random.randint(nsenna, size = (hp['nneg'],))
                nchunk = []
                for kk in range(hp['nneg']):
                    tmpchunk = copy.copy(pchunk)
                    tmpchunk[idxsta+nd] = rndidx[kk]
                    nchunk += tmpchunk
                assert len(nchunk) == len(pchunk)*hp['nneg']
                p, n  = (idx2mat(pchunk,nsenna), idx2mat(nchunk,nsenna))
                l = opt(p,n, idxsta, idxsto)
                c.append(l)

                if debug:
                    print '.',
                    break


            if debug:
                print ''

            bigc += [numpy.array(c).sum()]

            if 0:#(time.time() - expstart) > ( 3600 * 24 * 6 + 3600*20) or (tseenwords)>(10*hp['freq']):
                tseenwords = 0
                valid_embedding.params['weights'] = sp.shared(value = scipy.sparse.csr_matrix(embedding.params['e_weights'].get_value(borrow=True)))
                mrk = evaluation.error(validsentence, validfct, nsenna, hp['wsize'])
                hp['mrk'] = mrk
                jobman.save()
                saveexp()
                print 'Random Valid Mean rank',mrk


            if seen_words > hp['freq'] or debug:
                seen_words = 0
                hp['score'] = numpy.array(bigc).mean() 
                hp['e'] = e
                hp['i'] = i
                print ''
                print e,i,'NN Score:', hp['score']

                if not debug:
                    ne = knn(freq_idx,embedding.params['e_weights'].get_value(borrow=True))
                    open('files/'+fname+'nearest.txt','w').write(display(ne,senna))
                    saveexp()
                sys.stdout.flush()
                jobman.save()
                
    saveexp()

def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    #run(jobhandler,True)
    run(jobhandler,False)
    return 0


if __name__ == "__main__":
    HP_init = [ ('values','epoch',[100]),
                ('values','deviation',[.1]),
                ('values','iresume',[10000]),
                ('values','freq',[10000]),
                ('values','loadpath',[None]),
                ('values','hsize',[500,1000,2000]),
                ('values','embedsize',[50]),
                ('values','minsize',[5]),
                ('values','maxsize',[30]),
                ('values','wsize',[30]),
                ('values','npos',[1]),
                ('values','nneg',[10]),
                ('values','lr',[0.01,.005]),
                ('values','lambda',[.0]),
                ('values','margin',[1.]),
                ('values','bs',[10]) ]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)




from dataset.DatasetInterfaces import root

from util.embedding import knn, display
from util.cost import nll, hardmax, ce
from util.expr import rect, identity, hardtanh
from util.io import save, load
from util.sparse import idx2spmat, idx2mat, idx2vec

import evaluation
from unsupervised import cae,ae
from supervised import logistic
from jobscheduler import JobmanInterface as JB
from tools.hyperparams import hyperparams
import sparse.supervised
from test import wsim
import theano
import theano.sparse as sp
import theano.tensor as T
from theano import function
from theano.ifelse import ifelse

import scipy.sparse
import numpy
import sys,pdb,cPickle,os,copy
import time


def run(jobman,debug = False):
    expstart = time.time()
    hp = jobman.state

    if not os.path.exists('files/'): os.mkdir('files/')

    # Symbolic variables
    s_posit = T.matrix()
    s_negat = T.matrix()
    s_valid = theano.sparse.csr_matrix()


    w2i = cPickle.load(open('/mnt/scratch/bengio/bengio_group/data/gutenberg/merged_word2idx.pkl'))
    i2w = dict( (v,k) for k,v in w2i.iteritems() )
    i2w[0] = 'UNK'
    senna = [ i2w[i] for i in range(len(i2w.keys())) ]

    nsenna = len(senna)
    
    embedding = cae(i_size=nsenna, h_size=hp['embedsize'], e_act = identity)
    H = ae(i_size = hp['embedsize']*hp['wsize'], h_size=hp['hsize'], e_act = T.tanh)
    L = logistic(i_size = hp['hsize'], h_size = 1, act = identity)
 
    path = hp['loadpath']
 
    if path:
        load(embedding,path+'/embedding.pkl')
        load(H,path+'/hidden.pkl')
        load(L,path+'/logistic.pkl')
        hp['embedsize'] = embedding.params['e_weights'].get_value(borrow=True).shape[1]
        hp['hsize'] = H.params['e_weights'].get_value(borrow=True).shape[1]
        jobman.save()

    del H.params['d_bias']
    del embedding.params['d_bias']
    del embedding.params['e_bias']


    lr = hp['lr']
    h_size = hp['hsize']
    bs = hp['bs']

    posit_embed = theano.sparse.dot(s_valid, embedding.params['e_weights']).reshape((1+hp['nneg'],hp['embedsize']*hp['wsize']))
    #posit_embed = T.dot(s_valid, embedding.params['e_weights']).reshape((1+hp['nneg'],hp['embedsize']*hp['wsize']))
    #negat_embed = T.dot(s_negat, embedding.params['e_weights']).reshape((hp['nneg'],hp['embedsize']*hp['wsize']))
    
    posit_score = L.encode(H.encode(posit_embed))
    #negat_score = L.encode(H.encode(negat_embed))


    C = (posit_score[1:] - posit_score[0] + hp['margin'])

    CC = (rect(C)).mean()

    opt = theano.function([s_valid],
                          (rect(C)).mean(),
                          updates = dict( L.update(CC,lr) + H.update(CC,lr) + embedding.update_norm(CC,lr)) )


    def saveexp():
        save(embedding,fname+'embedding.pkl')
        save(H,fname+'hidden.pkl')
        save(L,fname+'logistic.pkl')


    delta = hp['wsize']/2
    rest = hp['wsize']%2

    wsimtester = wsim()
    freq_idx = cPickle.load(open('/mnt/scratch/bengio/bengio_group/data/gutenberg/sorted_vocab.pkl'))[:1000]
    
    fname = ''

    tseenwords = not debug
    for e in range(hp['epoch']):
        hp['split'] = numpy.random.randint(285)
        sentences = cPickle.load(open('/mnt/scratch/bengio/bengio_group/data/gutenberg/small_ints_50000/split'+str(hp['split'])+'.pkl'))
        nsent = len(sentences)
        bigc = []
        bigr = []
        E = embedding.params['e_weights'].get_value(borrow=True)
        seen_words = 0
        for i,s in enumerate(sentences):
            nword = len(s)
            seen_words += nword
            tseenwords += nword

            if nword < hp['wsize'] + 2:
                continue
            c =[]
            r =[]
            if debug:
                print ' *** Processing document',i,'with',nword,
                sys.stdout.flush()
            for j in range(delta,nword-delta):
                cstart =time.time()
                pchunk = s[j-delta:j+delta+rest]
                nchunk = []
                st = s[j-delta:j]
                en = s[j+1:j+delta+rest]
                rndidx = numpy.random.randint(nsenna, size = (hp['nneg'],))
                nchunk = []
                for kk in range(hp['nneg']):
                    nchunk += st + [rndidx[kk]] + en

                #assert len(nchunk) == len(pchunk)*hp['nneg']
                #p, n  = (idx2mat(pchunk,nsenna), idx2mat(nchunk,nsenna))
                pn = idx2spmat(pchunk+nchunk,nsenna)
                #assert pn[0,pchunk[0]] == 1
                ctime = time.time() - cstart
                tstart = time.time()

                l = opt(pn)
                c.append(l)

                if debug:
                    print '.',
                    break


            if debug:
                print ''

            bigc += [numpy.array(c).sum()]

            if  tseenwords > hp['wsimfreq'] or debug:
                hp['wsim'] = wsimtester.score(embedding.params['e_weights'].get_value(borrow=True))
                print i,'WordSim Sp Corr:',hp['wsim']
                sys.stdout.flush()
                hp['score'] = numpy.array(bigc).mean() 
                hp['e'] = e
                hp['i'] = i
                print e,i,'NN Score:', hp['score']
                tseenwords = 0
                jobman.save()

            if seen_words > hp['freq'] or debug:
                seen_words = 0
                ne = knn(freq_idx,embedding.params['e_weights'].get_value(borrow=True))
                open('files/'+fname+'nearest.txt','w').write(display(ne,i2w))
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
                ('values','freq',[20000]),
                ('values','wsimfreq',[2000]),
                ('values','loadpath',[None]),
                ('values','hsize',[100]),
                ('values','embedsize',[50]),
                ('values','wsize',[5,7]),
                ('values','npos',[1]),
                ('values','nneg',[10,30]),
                ('values','lr',[.01,.005,.001,.0005,.0001]),
                ('values','lambda',[.0]),
                ('values','margin',[1.]),
                ('values','bs',[10]) ]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)




import cPickle
import numpy
import sys
import os
import pdb
import theano
import theano.tensor as T

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
import subprocess



def idxdataset(vocab):
    voc = cPickle.load(open(vocab))
    dvoc = dict(zip(voc,range(len(voc))))
    path = '/scratch/rifaisal/msrtest/test/data/'
    answers = open(path+'Holmes.lm_format.answers.txt')
    questions = open(path+'Holmes.lm_format.questions.txt')
    cqlist = []
    for a in answers:
        for i in range(5):
            q = questions.readline()
            q = q.split('<s>')[-1]
            q = q.split('</s>')[0]
            cqlist.append(q.lower())#.replace("'", ' ').replace("-", ' '))
    sidx = []
    for s in cqlist:
        cidx = []
        #pdb.set_trace()
        for w in s.split(' ')[1:-1]:
            try:
                cidx.append(dvoc[w.lower()]) 
            except:
                print w.lower(),
                #pdb.set_trace()
                cidx.append(dvoc['UUUKKKNNN'])
        sidx.append(cidx)
    return sidx


def msrerror(vocab,jobman):
    hp = jobman.state
    nsenna = 30000

    PATH = '/scratch/rifaisal/msrtest/test/'
    delta = hp['wsize']/2
    rest = hp['wsize']%2
    sent = T.matrix()

    embedding = cae(i_size=nsenna, h_size=hp['embedsize'], e_act = identity)
    H = ae(i_size = hp['embedsize']*hp['wsize'], h_size=hp['hsize'], e_act = T.tanh)
    L = logistic(i_size = hp['hsize'], h_size = 1, act = identity)

    path = hp['loadpath']

    load(embedding,path+'/embedding.pkl')
    load(H,path+'/hidden.pkl')
    load(L,path+'/logistic.pkl')

    posit_embed = T.dot(sent, embedding.params['e_weights']).reshape((1,hp['embedsize']*hp['wsize']))
    posit_score = L.encode(H.encode(posit_embed))
    fct = theano.function([sent],posit_score)
    sentences = idxdataset(vocab)
    scores = []
    for i,s in enumerate(sentences):
        print i,
        sys.stdout.flush()
        nword = len(s)
        if nword < hp['wsize'] + 2:
            #print i,'Failure'
            s += [29999]*3
        c =[]
        for j in range(delta,nword-delta):
            pchunk = s[j-delta:j+delta+rest]
            p = idx2mat(pchunk,nsenna)
            l = fct(p)
            c.append(l)
        if not len(c):
            print 'pas bim'
            scores.append(0)
        else:
            scores.append(numpy.mean(c))
        #if i%5 == 0:
        #    print scores[i-5:i]

    score_template = open(PATH+'data/Holmes.lm_format.questions.txt')
    score_output = open('energy.lm_output.txt','w')
    sentencelist = score_template.readlines()
    for sc,sentence in zip(scores, sentencelist):
        score_output.write(sentence.split('\n')[0]+'\t'+str(sc)+'\n')
    score_output.close()

    pipebestof5 = subprocess.Popen(['perl', PATH+'bestof5.pl','./energy.lm_output.txt'],stdout=subprocess.PIPE)
    energyanswer = open('./energy.answers','w')

    for line in pipebestof5.stdout: energyanswer.write(line)

    energyanswer.close()

    pipescore = subprocess.Popen(['perl', PATH+'score.pl','./energy.answers',PATH+'data/Holmes.lm_format.answers.txt'],stdout=subprocess.PIPE)
    legend = ['correct','%correct','valid','test']
    out = zip(legend,[ r.split('\n')[0] for r in  pipescore.stdout.readlines()[-4:] ])
    res = dict(out)
    res = dict( (k,float(v)) for k,v in res.iteritems())
    print res
    print out

def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    msrerror('/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004_vocab30k.pkl',jobhandler)
    return 0


if __name__ == "__main__":
    HP_init = [ ('values','epoch',[100]),
                ('values','deviation',[.1]),
                ('values','iresume',[3756]),
                ('values','freq',[10000]),
                ('values','loadpath',['/scratch/rifaisal/exp/mullerx_db/wikibaseline_bugfix_0002/2/files']),
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






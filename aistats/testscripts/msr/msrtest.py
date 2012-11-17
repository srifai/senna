#!/usr/bin/env python

from jobscheduler import JobmanInterface as JB
from tools.hyperparams import hyperparams
from energy import linear,mlp
import theano
import theano.tensor as T
import numpy
import cPickle
import os
import pdb
import sys
import subprocess

def test(self):
    path = '/scratch/rifaisal/data/Holmes_data/'
    files = [ path+'MSR_scomp_holmes_questions_diff_trees_only_3.pkl' ]

    score = []

    for f in files:
        print 'Scoring',f
        sys.stdout.flush()
        test = cPickle.load(open(f))
        for fivesentence in test:
            for sentence in fivesentence:
                n = len(sentence)
                x = (sentence[:,:,:300*self.ngrams].reshape((-1,300*self.ngrams*17)) - self.mean)/ self.std
                score += [numpy.max(self.forward(x)[0])]
    assert len(score) == 1040*5

    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')
    #score = range(1040*5)
    PATH = '/scratch/rifaisal/msr/test/'
    score_template = open(PATH+'data/Holmes.lm_format.questions.txt')
    score_output = open('tmp/energy.lm_output.txt','w')
    sentencelist = score_template.readlines()
    for sc,sentence in zip(score, sentencelist):
        score_output.write(sentence.split('\n')[0]+'\t'+str(sc)+'\n')
    score_output.close()

    pipebestof5 = subprocess.Popen(['perl', PATH+'bestof5.pl','./tmp/energy.lm_output.txt'],stdout=subprocess.PIPE)
    energyanswer = open('./tmp/energy.answers','w')

    for line in pipebestof5.stdout: energyanswer.write(line)

    energyanswer.close()

    pipescore = subprocess.Popen(['perl', PATH+'score.pl','./tmp/energy.answers',PATH+'data/Holmes.lm_format.answers.txt'],stdout=subprocess.PIPE)
    legend = ['correct','%correct','valid','test']
    out = zip(legend,[ r.split('\n')[0] for r in  pipescore.stdout.readlines()[-4:] ])
    res = dict(out)
    res = dict( (k,float(v)) for k,v in res.iteritems())
    print res
    print out
    if self.jobman.state['valid'] < res['valid']:
        self.jobman.state['valid'] = res['valid']
        self.jobman.state['test'] = res['test']
        print 'bim!'
        #self.mlp.save('./files/')

    if self.jobman.state['btest'] < res['test']: self.jobman.state['btest'] = res['test']

    self.jobman.save()


def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    exp = experiment(jobhandler)
    exp.run()
    return 0


if __name__ == "__main__":
    HP_init = [ ('uniform','lr',[ .03, .02] ),
                ('values','n_h', [1000,2000]),
                ('values','bs', [100,100]),
                ('values','ngrams',[4,5,6,7]),
                ('values','epoch', [1000]),
                ('values','corruption',[10,2,3]),
                ('values','train_type',['pt'])]
    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)

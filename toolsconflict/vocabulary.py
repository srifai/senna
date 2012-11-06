import pdb
import numpy
import sys,os
import cPickle

class vocabulary(object):

    def __init__(self, wordlist, frequency=None):
        self.w2i = dict(zip(wordlist,range(len(wordlist))))
        self.i2w = dict(zip(range(len(wordlist)),wordlist))

    @classmethod
    def FromFile(vocab,f):
        a = [[str(ll[0]),int(ll[1]),int(ll[2])] for ll in [ l.split('\n')[0].split(' ') for l in open('trainEmb/data/vocab.txt').readlines() ] ]
        pdb.set_trace()
        return vocab(a)




def wiki(f='/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004.txt'):
    reader = open(f,'r')
    docs = []
    doc = ''
    for line in reader:
        print '.',
        if "---END.OF.DOCUMENT---" not in line:
            doc += line
        else:
            print ''
            docs.append(doc)
            doc = ''



    cPickle.dump(docs,open('/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004.pkl','w'))

def wiki_es(f='/scratch/rifaisal/data/wiki_es/extracted/AA/wiki_00'):
    reader = open(f,'r')
    docs = []
    doc = ''
    for line in reader:
        print '.',
        if "<doc" not in line:
            doc += line
        else:
            print ''
            docs.append(doc)
            doc = ''



    cPickle.dump(docs,open('/scratch/rifaisal/data/wiki_es/wiki_espanol_list.pkl','w'))


if __name__ == '__main__':
    pass

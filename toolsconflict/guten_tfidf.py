import numpy
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.feature_extraction.text import CountVectorizer as cvect
from nltk import WordPunctTokenizer as wt

import pdb
import operator
import unicodedata
import sys
import string,re

def build_guten():
    sentences = cPickle.load(open('/scratch/rifaisal/data/guten/guten_subset_idx.pkl'))

    senna = cPickle.load(open('/scratch/rifaisal/data/guten/senna.pkl'))
    gsubset = cPickle.load(open('/scratch/rifaisal/data/guten/guten_vocab_subset.pkl')).flatten().tolist()
    
    senna = numpy.array(senna)[gsubset].tolist()
    hashtab = dict( zip( senna, range( len( gsubset))))


    vectorizer = tfidf(vocabulary=hashtab,stop_words='english')

    wsentences = []
    avglen = 0
    for s in sentences:
        print '*',
        news = ''
        for w in s:
            print '.',
            news +=' ' + senna[w]
        avglen += len(s)
        wsentences.append(news)
        print ''

    print 'Average sentence size:',avglen/float(len(sentences))
    tfidfmat = vectorizer.fit_transform(wsentences)
    numpy.save('/scratch/rifaisal/data/guten/guten_tfidf.npy',tfidfmat)
    cPickle.dump(vectorizer,open('gutentokenizer.pkl','w'))
    print 'Done!'

def build_wiki_vocab(corpus):
    sentences = corpus

    tokenizer = wt()
    totalvocab = {}

    for s in sentences:
        strlist = tokenizer.tokenize(s.lower())
        for w in strlist:
            try:
                totalvocab[w] +=1
            except: 
               totalvocab[w] = 1

    sortvoc = sorted(totalvocab.iteritems(), key=operator.itemgetter(1))
    return sortvoc


def remove_accents(data):
    return unicodedata.normalize('NFKD', unicode(data)).encode('ascii', 'ignore')


def clean_vocab(vocab):
    new = []
    import re, string
    pattern = re.compile('[\W_]+')
    for w,f in vocab:
        try:
            w = remove_accents(w)
        except:
            continue

        if len(w.split("\\x")) > 1:
            continue
        if len(w) <= 1 and w not in string.punctuation:
            continue

        if len(re.sub(r'\W+', '', w)) == 0 and len(w) != 1:
            continue

        new += [(w,f)]
    return new

def build_wiki_data(corpus, vocab , k= 30000):
    mostfreqk = vocab[-k+5:]
    vocabd = dict(zip( [w for w,v in mostfreqk ], range(k-5)))
    
    vocabd['DG'] = k-5
    vocabd['DGDG'] = k-4
    vocabd['DGDGDG'] = k-3
    vocabd['DGDGDGDG'] = k-2
    vocabd['UUUKKKNNN'] = k-1

    assert(len(vocabd.keys()) == k )

    newcorpus = []
    newcorpusf = []
    tokenizer = wt()
    print 'Total documents:',len(corpus)
    for aa,doc in enumerate(corpus):
        print aa,
        sys.stdout.flush()
        strlist = tokenizer.tokenize(doc.lower())
        docidx = []
        docfiltered = []
        for w in strlist:
            try:
                docidx.append(vocabd[w])
                #docfiltered.append(w)
            except:
                if w.isdigit() and len(w) <= 4:
                    docidx.append(k-len(w)-1)
                    #docfiltered.append('DG'*len(w))
                    continue
                else:

                    try:
                        w = remove_accents(w)
                    except:
                        continue

                    if len(w.split("\\x")) > 1:
                        continue
                    elif len(w) <= 1 and w not in string.punctuation:
                        continue

                    elif len(re.sub(r'\W+', '', w)) == 0 and len(w) != 1:
                        continue
                    else:
                        docidx.append(k-1)
                        #docfiltered.append('UUUKKKNNN')


        newcorpus.append(docidx)
        #newcorpusf.append(docfiltered)

    return newcorpus, newcorpusf, vocabd

def build_wiki_data_es(corpus, vocab , k= 30000):
    mostfreqk = vocab[-k+5:]
    vocabd = dict(zip( [w for w,v in mostfreqk ], range(k-5)))
    
    vocabd['DG'] = k-5
    vocabd['DGDG'] = k-4
    vocabd['DGDGDG'] = k-3
    vocabd['DGDGDGDG'] = k-2
    vocabd['UUUKKKNNN'] = k-1

    assert(len(vocabd.keys()) == k )

    newcorpus = []
    newcorpusf = []
    tokenizer = wt()
    print 'Total documents:',len(corpus)
    for aa,doc in enumerate(corpus):
        print aa,
        sys.stdout.flush()
        strlist = tokenizer.tokenize(doc.lower())
        docidx = []
        docfiltered = []
        for w in strlist:
            try:
                docidx.append(vocabd[w])
                #docfiltered.append(w)
            except:
                if w.isdigit() and len(w) <= 4:
                    docidx.append(k-len(w)-1)
                    #docfiltered.append('DG'*len(w))
                    continue
                else:

                    try:
                        w = remove_accents(w)
                    except:
                        continue

                    if len(w.split("\\x")) > 1:
                        continue
                    elif len(w) <= 1 and w not in string.punctuation:
                        continue

                    elif len(re.sub(r'\W+', '', w)) == 0 and len(w) != 1:
                        continue
                    else:
                        docidx.append(k-1)
                        #docfiltered.append('UUUKKKNNN')


        newcorpus.append(docidx)
        #newcorpusf.append(docfiltered)

    return newcorpus, newcorpusf, vocabd


if __name__ == "__main__":
    #vocab = build_wiki_vocab(cPickle.load(open('/scratch/rifaisal/data/wiki_es/wiki_espanol_list.pkl')))
    #pdb.set_trace()
    vocab = cPickle.load(open('/scratch/rifaisal/data/wiki_es/wiki_es_fullvocab.pkl'))
    corpus = cPickle.load(open('/scratch/rifaisal/data/wiki_es/wiki_espanol_list.pkl'))
    idx, words, voc = build_wiki_data(corpus,vocab)
    wikifile = open('/scratch/rifaisal/data/wiki_es/wiki_es_prepro30k.pkl','w')
    cPickle.dump(idx,wikifile)
    cPickle.dump(words,wikifile)
    cPickle.dump(voc,wikifile)
    wikifile.close()

    """
    corpus = cPickle.load(open('/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004.pkl'))
    vocab = cPickle.load(open('wikicompletevocab.pkl'))
    #pdb.set_trace()
    idx, words, voc = build_wiki_data(corpus,vocab)
    wikifile = open('/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004_prepro30k.pkl','w')
    cPickle.dump(idx,wikifile)
    cPickle.dump(words,wikifile)
    cPickle.dump(voc,wikifile)
    wikifile.close()
    """

import numpy
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import pdb
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

def build_wiki():

    sentences = cPickle.load(open('/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004.pkl'))
    senna = cPickle.load(open('/scratch/rifaisal/data/guten/senna.pkl'))
    gsubset = cPickle.load(open('/scratch/rifaisal/data/guten/guten_vocab_subset.pkl')).flatten().tolist()
    
    senna = numpy.array(senna)[gsubset].tolist()
    hashtab = dict( zip( senna, range( len( gsubset))))


    vectorizer = tfidf(vocabulary=hashtab,stop_words='english')

    tfidfmat = vectorizer.fit(sentences[:len(sentences)/2])
    bs = len(sentences)/4
    for i in range(4):
        numpy.save('/scratch/rifaisal/data/wiki_april_2010/wiki_tfidf_'+str(i)+'.npy',vectorizer.transform(sentences[i*bs:(i+1)*bs]))

    cPickle.dump(vectorizer,open('wikitokenizer.pkl','w'))
    print 'Done!'



if __name__ == "__main__":
    build_wiki()

import nltk
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk import load

def loadLexicon(fname):
    
    newLex=set()
    lex_conn=open(fname)
    
    for line in lex_conn:
        newLex.add(line.strip())
    lex_conn.close()
    
    return newLex


def getPOSterms(terms, POStags, tagger):
    tagged_terms = tagger.tag(terms)
    POSterms = {}
    for tag in POStags:
        POSterms[tag] = set()
        for pair in tagged_terms:
            for tag in POStags:
                if pair[1].startswith(tag): 
                    POSterms[tag].add(pair[0])
    return POSterms


def processSentence(sentence,posLex,negLex,tagger):  
    
    terms = nltk.word_tokenize(sentence.lower())
    
    POStags = ['NN']
    POSterms = getPOSterms(terms, POStags, tagger)
    nouns = POSterms['NN']
    
    result=[]
    fourgrams = ngrams(terms,4) #compute 4-grams    
   	 #for each 4gram
    for tag in fourgrams:  
        if tag[0] =='not' and (tag[2] in posLex or tag[2] in negLex) and tag[3] in nouns: 
            result.append(tag)
    
    return result
    
def run(fpath):
    
    # Pos and neg list
    posLex = loadLexicon('positive-words.txt')
    negLex = loadLexicon('negative-words.txt')
    
    # tagger
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagger = load(_POS_TAGGER)
    
    #read the input
    f=open(fpath)
    text=f.read().strip()
    f.close()

    #split sentences
    sentences=sent_tokenize(text)
    print ('NUMBER OF SENTENCES: ',len(sentences))
    
    notPosNegNoun = []
    
    for sentence in sentences:
        
        notPosNegNoun+=processSentence(sentence, posLex, negLex, tagger)
    
    return notPosNegNoun

    
if __name__ == "__main__":
    print(run('input.txt'))
    
    
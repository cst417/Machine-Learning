#Examples from Applied Text Analysis with Python
import nltk
import string
from collections import defaultdict


def tokenize(text):
    """
    Tokenize method will remove punctuation, put text to lowercase and reduce features using the SnowballStemmer (eg bats become bat etc.)
    """
    stem = nltk.stem.SnowballStemmer("english")
    text=text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)

corpus = [
    "The elephant sneezed at the sight of potatoes",
    "Bats can see via echolocation. See the bat sight sneeze!?@",
    "WOndering, she opened the door to the studio"
]

"""
Frequency vectors
"""
#Using NLTK approach

def vectorize(doc):
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1
    return features

vectors = map(vectorize, corpus)

#Scikit-learn approach
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(corpus)

#note: If the corpus is huge, use HashingVectoizer instead. It uses hashing and speeds up the process. There is no inverse transform
#from vector to text, hash collisions can happen and no inverse document frequence weighting.

"""
One Hot Encoding
"""

#NLTK approach 
def vectorize_OHE(doc):
    return{
        token: True 
        for token in doc
    }

#Scikit learn
from sklearn.preprocessing import Binarizer

freq = CountVectorizer()
corpus = freq.fit_transform(corpus)


onehot = Binarizer()
corpus = onehot.fit_transform(corpus.toarray())     #toarray() is ideal when corpora is small, since it converts sparse matrix into a 
#dense one. If the corpora is huge, then a sparse matrix is prioritized.

"Term Frequency-Inverse Document Frequency"

#NLTK
from nltk.text import TextCollection

def vectorize_TFIDF(corpus):
    corpus = [tokenize(doc) for doc in corpus] #applying tokenization
    texts = TextCollection(corpus) #creating text collection
#goes through each document in corpus and yields a dictionary whos keys are terms and values are the score given by the TF-IDF
    for doc in corpus:
        yield {
            term: texts.tf_idf(term, doc)
            for term in doc 
        }

#Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
corpus = tfidf.fit_transform(corpus)

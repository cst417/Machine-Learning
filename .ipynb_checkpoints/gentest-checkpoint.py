from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import gensim
import nltk
import string

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

corpus = [list(tokenize(doc)) for doc in corpus]
corpus = [

]
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd 
import nltk
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import webbrowser as rowreader
from xgboost import XGBClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.preprocessing import FunctionTransformer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def create_pipeline(estimator):
    steps = [
            ('vectorizer', CountVectorizer(tokenizer= LemmaTokenizer(),
                                            stop_words='english',
                                            ngram_range=(2,2),
                                            lowercase=True)),
            ('transformer', FunctionTransformer(lambda x: x.todense(), accept_sparse=True))
    ]

    steps.append(('classifier', estimator))
    return Pipeline(steps)

nltk.download("stopwords")
file = "movie_review.csv"
col_list = ['text', 'tag', 'datz']

read = pd.read_csv(file, sep=',', usecols=col_list)

X = read['text']
y = read['tag']
z = read['datz']

if z is not np.NaN:
    rowreader.open(z[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=632)

models = []
models.append(create_pipeline(GaussianNB()))
models.append(create_pipeline(MultinomialNB()))

for model in models:
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))





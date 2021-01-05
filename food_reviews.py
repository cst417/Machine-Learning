import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import classification_report
import numpy as np 
col_list = ['text', 'tag']
data = pd.read_csv("movie_review.csv", usecols=col_list)
#nltk.download('stopwords')
corpus =[]
for i in range(0, 64720):
    review = re.sub('[^a-zA-Z]',' ', data['text'][i]) #keeps a-z and A-Z, replaces useless stuff with space.
    review = review.lower() 
    review = review.split() #easier to clean when separated 
    ws = WordNetLemmatizer()
    review = [ws.lemmatize(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review)
    corpus.append(review)

# print(corpus)
# # Cleaning the texts for 1 review
# review = re.sub('[^a-zA-Z]',' ', data['Review'][0]) #keeps a-z and A-Z, replaces useless stuff with space.
# review = review.lower() 
# # print(review)
# # Get rid of irrelevant words
# # nltk.download('stopwords')
# #split the review into separate words
# review = review.split() #easier to clean when separated 
# ps = PorterStemmer()
# review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  #set makes it faster, ps.stem stemms every word and puts them in the review
# # reverting review back to a sentence
# review = ' '.join(review)
# print(review)

#creating bag of words model 
vec = CountVectorizer(max_features= 1500)
X = vec.fit_transform(corpus).toarray()
# print(X)
y = data['tag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# print("X_train:", X_train)
# print("X_test",X_test)
"""
naiveb = GaussianNB()
naiveb.fit(X_train,y_train)
y_pred = naiveb.predict(X_test)
matrix = classification_report(y_test, y_pred)
print(matrix)
"""
models = []
#models.append(GaussianNB())
models.append(SVC(C=1.0, kernel='rbf'))

for model in models:
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))
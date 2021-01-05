#https://machinelearningmastery.com/calculate-feature-importance-with-python/

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)

model = DecisionTreeClassifier()

model.fit(X, y)

importance = model.feature_importances_

for i, v in enumerate(importance):
    print ('Feature %0d, Score: %0f'%(i, v))

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
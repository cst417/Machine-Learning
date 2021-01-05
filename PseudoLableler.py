#https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/
import pandas as pd 
from sklearn.utils import shuffle   
from sklearn.base import BaseEstimator, RegressorMixin

class PesudoLabeler(BaseEstimator, RegressorMixin):
    def __init__(self, model, unlabeled_data, features, target, sample_rate=0.2, seed=40):
        assert sample_rate <= 1.0
        self.model = model
        self.seed = seed
        self.features = features 
        self.target = target
        self.unlabeled_data = unlabeled_data
        self.sample_rate = sample_rate
    
    def get_params(self, deep=True):
         return {
                "sample_rate": self.sample_rate,
                "seed": self.seed,
                "model": self.model,
                "unlabled_data": self.unlabeled_data,
                "features": self.features,
                "target": self.target
                }
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            return self
    def fit(self, X, y):
        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
        augemented_train[self.features],
        augemented_train[self.target]
        )

        return self


    def __create_augmented_train(self, X, y):
        num_of_samples = int(len(self.unlabeled_data) * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabeled_data[self.features])

        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabeled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)

    def predict(self, X):

        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__

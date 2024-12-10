import numpy as np
from baseclasses.baseregression import BaseRegression

class LogisticRegression(BaseRegression):

    def _approximation(self, X, w, b):
        linear_model =  np.dot(X,w) + b
        return self.__sigmoid(linear_model)

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self.__sigmoid(linear_model)
        y_predicted_cls = [1 if val > 0.5 else 0 for val in y_predicted]
        return y_predicted_cls

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
import numpy as np
from baseclasses.baseregression import BaseRegression

class LinearRegression(BaseRegression):

    def _approximation(self, X, w, b):
        return np.dot(X,w) + b

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
        
    def _predict(self, X, w, b):
        return np.dot(X,w) + b
    
    
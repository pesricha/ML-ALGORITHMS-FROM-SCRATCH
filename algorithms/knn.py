import numpy as np
from collections import Counter


class KNN:
    

    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self.__predict(x) for x in X]
        return np.array(predicted_labels)

    def __predict(self, x):
        # compute distance from each training point
        distances = [np.linalg.norm(x=x_train - x) for x_train in self.X_train]

        # get k Nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices] 

        # do a majority vote and return it
        most_common_label = Counter(k_nearest_labels).most_common(1)
        return most_common_label[0][0]


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from algorithms.decision_tree import DecisionTree

def accuracy(y_pred, y_true):
    return np.sum(y_true == y_pred) / len(y_pred)

data =  datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=1234)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accu = accuracy(y_pred, y_test)

print(f"Accuracy : {accu}")
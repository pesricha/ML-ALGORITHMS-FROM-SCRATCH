import numpy as np
from algorithms.decision_tree import DecisionTree
from collections import Counter

def bootstrap_sample(X, y):
    """
    Generate a bootstrap sample from the given dataset.
    Parameters:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
    y (numpy.ndarray): Target vector of shape (n_samples,).
    Returns:
    tuple: A tuple containing the bootstrap sample of the feature matrix and the target vector.
    """

    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

def most_common_label(y):
    ctr = Counter(y)
    most_common = ctr.most_common(1)[0][0]
    return most_common

class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # We will swapaxes to get presictions from differemnt trees in one fibre
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = np.array([most_common_label(tree_pred) for tree_pred in tree_preds])
        return y_pred

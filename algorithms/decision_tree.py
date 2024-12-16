import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return True if self.value is not None else False
        

class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_sample_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the decision tree.
        Parameters:
        -----------
        X : ndarray
            Feature matrix of shape (n_samples, n_features).
        y : ndarray
            Target vector of shape (n_samples,).
        depth : int, optional (default=0)
            Current depth of the tree.
        Returns:
        --------
        Node
            The root node of the decision tree.
        """
        
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Base case
        if (
            depth >= self.max_depth or
            n_labels==1 or
            n_samples < self.min_sample_split
        ):
           leaf_value = self._most_common_label(y)
           return Node(value=leaf_value)
        
        feature_indices = np.random.choice(n_features, self.n_feats, replace= False)

        # Greedy Search
        best_feature, best_threshold = self._best_criteria(X, y, feature_indices)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        # Return the Node
        return Node(best_feature, best_threshold, left, right)

    def _best_criteria(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split the data.
        Parameters:
        X (numpy.ndarray): The input features of shape (n_samples, n_features).
        y (numpy.ndarray): The target values of shape (n_samples,).
        feature_indices (list): List of feature indices to consider for splitting.
        Returns:
        tuple: The index of the best feature to split on and the best threshold value.
        """
        
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feature_index, threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        """
        Calculate the information gain of a potential split in the decision tree.
        Parameters:
        y (array-like): Target values.
        X_column (array-like): Feature values for the split.
        split_thresh (float): Threshold value to split the feature.
        Returns:
        float: Information gain of the split.
        """
        # parent entropy
        parent_entropy = entropy(y)
        # generate split 
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # weighted avg of children 
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        #return IG
        ig = parent_entropy - child_entropy
        return ig
    
    def _split(self, X_col, split_thresh):
        """
        Splits the data into two groups based on the given threshold.

        Parameters:
        X_col (numpy.ndarray): The column of data to split.
        split_thresh (float): The threshold value to split the data.

        Returns:
        tuple: Two numpy arrays, the first containing the indices of the elements 
               less than or equal to the threshold, and the second containing the 
               indices of the elements greater than the threshold.
        """

        left_idxs = np.argwhere(X_col <= split_thresh).flatten()
        right_idxs = np.argwhere(X_col > split_thresh).flatten()

        return left_idxs, right_idxs

    def _most_common_label(self, y):
        """
        Determine the most common label in the given list of labels.
        Parameters:
        y (list): A list of labels.
        Returns:
        The most common label in the list.
        """
        counter_objext = Counter(y)
        most_common = counter_objext.most_common(1)[0][0]
        return most_common
    
    def _traverse_tree(self, x, node:Node):
        """
        Traverse the decision tree to make a prediction for a single sample.
        Args:
            x (array-like): The input sample for which to make a prediction.
            node (Node): The current node in the decision tree.
        Returns:
            The value of the leaf node, which is the predicted class or value.
        """

        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    

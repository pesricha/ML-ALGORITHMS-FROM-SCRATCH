import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # calculate mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # calculate covariance (Here the function is defined wher features are rows instead of cols)
        cov = np.cov(X.T)

        # Eigen value decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        # sort and store only first n_components of eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[idxs]
        eigenvalues = eigenvalues[idxs]
        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        # project data onto space spanned by first n components
        X = X - self.mean
        return (self.components @ X.T).T
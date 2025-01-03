import numpy as np

class NaiveBayes:

    def fit(self, X:np.array, y:np.array):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._variance[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x,):
        mean = self._mean[class_idx]
        variance = self._variance[class_idx]
        numerator = np.exp(-((x-mean)**2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
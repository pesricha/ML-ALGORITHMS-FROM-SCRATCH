import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_i = y_[idx]
                condition = y_i * (np.dot(x_i, self.w) - self.b) >= 1
                dw = 2*self.lambda_param*self.w if condition else 2*self.lambda_param*self.w - np.dot(y_i, x_i)
                db = 0 if condition else y_i

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        linear_output = np.array([self._predict(x) for x in X])
        return linear_output

    def _predict(self, x):
        output = np.dot(self.w, x) + self.b
        return np.sign(output)

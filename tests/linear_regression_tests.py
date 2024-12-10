import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# print(X_train.shape)
# print(y_train.shape)

from algorithms.linear_regression import LinearRegression

regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
predicted_values = regressor.predict(X_test)
y_line_predict = regressor.predict(X)

plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color="r", marker='o', s=30)
plt.plot(X, y_line_predict, color='black', linewidth=2, label="Prediction")
plt.show()

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mse(y_test, predicted_values)
print(mse_value)
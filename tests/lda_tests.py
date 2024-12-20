import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from algorithms.lda import LDA

data = datasets.load_iris()
X, y = data.data, data.target

lda = LDA(2)
lda.fit(X, y)

X_transformed = lda.transform(X)

print(f"Shape of X : {X.shape}")
print(f"Shape of transformed X : {X_transformed.shape}")

x1 = X_transformed[:, 0]
x2 = X_transformed[:, 1]

plt.scatter(
    x1, x2, c=y, alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3)
)

plt.show()

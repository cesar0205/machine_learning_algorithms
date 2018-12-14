from unsupervised_learning.clustering.kmeans import Kmeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#Test the kmeans algorithm using 100 samples taken from 3 blobs.

X, y = make_blobs(n_samples=100, random_state=42)
model = Kmeans(init='kmeans++')
model.fit(X)
plt.title("Kmeans on 100 points and 3 centroids")
plt.scatter(X[:, 0], X[:, 1], c = model.labels_)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
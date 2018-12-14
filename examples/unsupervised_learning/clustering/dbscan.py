from unsupervised_learning.clustering.dbscan import DBScan
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(n_samples=100, cluster_std =2, random_state=42)
model = DBScan(min_samples = 5, eps= 2)
model.fit(X)


core_mask = np.zeros(len(X), dtype = bool)
core_mask[model.core_ind_] = True;
noise_mask = model.labels_ == -1;
non_core_mask = ~(core_mask | noise_mask)

plt.title("DBScan algorithm on 100 datapoints from 3 blobs")
#Print the core indexes
plt.scatter(X[core_mask, 0], X[core_mask, 1], c = model.labels_[core_mask], marker='x')
#Print the noise indexes
plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c = 'red', marker='+', s = 300)
#Print the non core indexes
plt.scatter(X[non_core_mask, 0], X[non_core_mask, 1], c = model.labels_[non_core_mask], marker='o')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
from unsupervised_learning.clustering.kernels import GaussianKernel
from sklearn.neighbors import NearestNeighbors
import numpy as np

'''
    Computes affinity matrix using an adjancy matrix with the nearest n_neighbors for each datapoint.
'''
class KnnAffinity():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors;

    def compute(self, X):
        model = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X);
        return model.kneighbors_graph(X).toarray();

'''
    Computes affinity matrix using a radial basis function kernel (such a gaussian kernel) to compute
    the similarity between points.
'''
class BasicAffinity():
    def __init__(self, kernel=GaussianKernel()):
        self.kernel = kernel;

    def compute(self, X):
        m = len(X)
        return np.array([self.kernel.dot(X[i], X[j])
                         for j in range(m) for i in range(m)]).reshape(m, m);

'''
    Computes affinity matrix using a gaussian kernel. The sigma1 and sigma2 are adaptative. They are being 
    calculated with the closest neighbors for each point.
'''
class LocalScaling():
    def __init__(self):
        self.kernel = GaussianKernel();

    def compute(self, X):
        m = len(X)
        dists = np.array([np.linalg.norm(X[i] - X[j])
                          for j in range(m) for i in range(m)]).reshape(m, m);
        dists = np.sort(dists);
        sig = dists[:, :5].mean(axis=1);
        return np.array([self.kernel.dot(X[i], X[j], sig[i], sig[j])
                         for j in range(m) for i in range(m)]).reshape(m, m);
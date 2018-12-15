from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs
from unsupervised_learning.clustering.affinities import BasicAffinity, KnnAffinity, LocalScaling
import numpy as np

'''
    Clusters data using the spectral clustering algorithm. Useful for data with complex structures or unknown shapes.
    https://calculatedcontent.com/2012/10/09/spectral-clustering/
'''
class SpectralClustering():
    def __init__(self, n_clusters=3, affinity=BasicAffinity):
        self.n_clusters = n_clusters;
        self.affinity = affinity;
        #Final clusterer.
        self.clusterer = KMeans(n_clusters=self.n_clusters);

    #Calculate the laplacian of a symmetric matrix
    def laplacian(self, W):
        d = np.sum(W, axis=0);
        D = np.diag(d ** (-0.5));
        return D.dot(W).dot(D);

    def fit(self, X):
        #1. Compute the affinity matrix
        A = self.affinity.compute(X);
        #2. Compute the laplacian of the affinity matrix
        L = self.laplacian(A);
        #3. Calculate the k largest eigenvectors of the laplacian.
        eig_val, eig_vec = eigs(L, k=self.n_clusters);
        X = eig_vec.real;
        #4.Normalize the eigenvectors by their norm over the last axis.
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        #5.Fit any clusterer to the new set.
        self.labels_ = self.clusterer.fit(X).labels_
        return self;

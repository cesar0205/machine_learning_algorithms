import numpy as np

class Kmeans():
    '''
        Clusters data using the K-means algorithm.
    '''
    def __init__(self, n_clusters=3, max_iterations=2000, init=None):
        self.n_clusters = n_clusters;
        self.max_iterations = max_iterations;
        self.init = None;

    def _init_centroids(self, X):
        '''
        We have two ways to initialize the initial centroids.
        Default initialization: Chooses the centroids by selecting K random points.
        kmeans++ initialization: Chooses the centroids so that they are spread out as much as possible.
        :param X: Data to cluster
        :return: Inicial centroids
        '''
        m, n = X.shape;
        if (self.init is None):
            rand_ind = np.random.choice(m, self.n_clusters, replace=False);
            return X[rand_ind];
        elif (self.init == 'Kmeans++'):
            centroids = np.full((self.n_clusters, n), np.inf);
            rand_ind = np.random.choice(m);
            centroids[0] = X[rand_ind];
            for i in range(1, self.n_clusters):
                dists = np.array([np.linalg.norm(x - centroids[self._closest_centroid(x, centroids)]) for x in X]);
                norm = dists / np.sum(dists);
                rand_ind = np.random.choice(m, p=norm);
                centroids[i] = X[rand_ind];
            return centroids[i];

    #Given the clusters calculate the new centroids.
    def _calculate_centroids(self, X, clusters):
        return np.array([X[cluster].mean(axis=0) for cluster in clusters]);

    #Given the centroids calculate the new clusters.
    def _create_clusters(self, X, centroids):
        clusters = [[] for i in range(self.n_clusters)];

        for x_ind, x in enumerate(X):
            closest_centroid = self._closest_centroid(x, centroids);
            clusters[closest_centroid].append(x_ind);

        return clusters;

    #Finds the centroid that is closest to x.
    def _closest_centroid(self, x, centroids):
        return np.argmin(np.linalg.norm(x - centroids, axis=1))

    #Calculates the distance between each point in X with all the centroids.
    def transform(self, X):
        return np.array([np.linalg.norm(x - self.centroids_, axis=1) for x in X]);

    def fit(self, X):
        m = len(X);
        centroids = self._init_centroids(X);
        for i in range(self.max_iterations):
            prev_centroids = centroids;
            clusters = self._create_clusters(X, prev_centroids);
            centroids = self._calculate_centroids(X, clusters);
            if not (centroids - prev_centroids).any():
                break;

        self.centroids_ = centroids;
        clusters = self._create_clusters(X, centroids)
        self.labels_ = np.zeros(m, int);
        for cluster_ind, cluster in enumerate(clusters):
            for x_ind in cluster:
                self.labels_[x_ind] = cluster_ind;

        dists = self.transform(X);

        #The inertia is a way to measure the performance of a clustering algorithm.
        #It is the squared sum of the distances of each point to its corresponding cluster.
        self.inertia_ = np.sum(dists[range(m), self.labels_] ** 2)
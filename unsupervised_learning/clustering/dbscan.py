import numpy as np

'''
    Clusters the data using the DBScan algorithm.
    The implementation is recursive. In the main loop we probe each datapoint and gather its neighbours (expand the 
    cluster) recursively until there are not more within a eps distance and they meet a minimum number of neighbours. 
    If a neighbor is already assigned to a cluster we probe the next one.
    The algorithm yields three types of datapoints. 
    The core indexes: Have at least min_samples neighbors.
    The non core indexes: Don't have at least min samples neighbors but at least 1.
    The noise indexes. They don't have any neighbors.
'''
class DBScan():
    def __init__(self, min_samples=5, eps=1e-3):
        '''
        Initialization
        :param min_samples: Minium number of neighbors to consider further expansion.
        :param eps: Minimum distance to consider a datapoint a neighbour.
        '''
        self.min_samples = min_samples;
        self.eps = eps;

    #Gets the neighbours of a datapoint.
    def _get_neighbors(self, x_ind):
        indexes = np.delete(range(self.m), x_ind);
        dists = np.linalg.norm(self.X[x_ind] - self.X[indexes], axis=1);
        return indexes[dists <= self.eps];

    #Expand the cluster starting from point x_ind to its neighbours n_inds.
    def _expand_cluster(self, x_ind, n_inds):

        cluster = [x_ind];

        for n_ind in n_inds:

            if (n_ind in self.visited_):
                continue;

            self.visited_.append(n_ind);

            n_n_inds = self._get_neighbors(n_ind);

            if (len(n_n_inds) >= self.min_samples):
                self.core_ind_.append(n_ind);
                cluster += self._expand_cluster(n_ind, n_n_inds);
            else:
                cluster.append(n_ind);

        return cluster;

    # Fits the dbscan model for the dataset X.
    # As an aside we keep track of all indices that have at least min_samples indexes and store them
    # in the list core_ind_.
    def fit(self, X):

        self.X = X;
        self.m = len(X);
        self.core_ind_ = [];
        self.visited_ = [];
        clusters = []
        for x_ind in range(self.m):

            if (x_ind in self.visited_):
                continue;

            self.visited_.append(x_ind);

            n_inds = self._get_neighbors(x_ind);

            if (len(n_inds) >= self.min_samples):
                self.core_ind_.append(x_ind);
                cluster = self._expand_cluster(x_ind, n_inds);
                clusters.append(cluster);

        #Store the core indexes: Datapoints with at least min_samples neighbours.
        self.core_ind_ = np.array(sorted(self.core_ind_));
        #Assign each datapoint in X to a cluster.
        self.labels_ = np.full(self.m, -1);
        for cluster_ind, cluster in enumerate(clusters):
            for x_ind in cluster:
                self.labels_[x_ind] = cluster_ind;

        return self;


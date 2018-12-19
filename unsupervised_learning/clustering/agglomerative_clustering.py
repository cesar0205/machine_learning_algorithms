import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from itertools import combinations
from scipy.spatial.distance import euclidean
import copy

'''
    Abstract class that different linkeage implementations will use to link two clusters.
'''
class AbstractLinkage():

    def alpha_fn(self, ni, nj, nk):
        raise NotImplementedError();


    def beta_fn(self, ni, nj, nk):
        raise NotImplementedError();


    def gamma_fn(self, ni, nj, nk):
        raise NotImplementedError();

    #Links two clusters given indexes i, j. Updates the distance matrix eliminating cluster j and updates the distance
    #between the new cluster with the rest of the clusters.
    def link(self, dist_matrix, dendrogram, i, j):
        m = len(dist_matrix);
        for k in range(m):
            if k != i and k != j:
                distance = self.compute_distance(dist_matrix, dendrogram, i, j, k);
                dist_matrix[i, k] = distance;
                dist_matrix[k, i] = distance;

        indexes = np.delete(range(m), j);
        return dist_matrix.take(indexes, axis=0).take(indexes, axis=1);


    def compute_distance(self, dist_matrix, dendrogram, i, j, k):
        ni = len(dendrogram[i]);
        nj = len(dendrogram[j]);
        nk = len(dendrogram[k]);

        alpha_i = self.alpha_fn(ni, nj, nk);
        alpha_j = self.alpha_fn(nj, ni, nk);
        beta = self.beta_fn(ni, nj, nk);
        gamma = self.gamma_fn(ni, nj, nk);

        d_ik = dist_matrix[i, k];
        d_jk = dist_matrix[j, k];
        d_ij = dist_matrix[i, j];

        return alpha_i * d_ik + alpha_j * d_jk + beta * d_ij + gamma * np.abs(d_ik - d_jk);

#Single linkage parameter definition
class SingleLinkage(AbstractLinkage):
    def alpha_fn(self, ni, nj, nk):
        return 0.5

    def beta_fn(self, ni, nj, nk):
        return 0;

    def gamma_fn(self, ni, nj, nk):
        return 0.5

#Complete linkage parameter definition
class CompleteLinkage(AbstractLinkage):
    def alpha_fn(self, ni, nj, nk):
        return 0.5

    def beta_fn(self, ni, nj, nk):
        return 0;

    def gamma_fn(self, ni, nj, nk):
        return -0.5

#Ward linkage parameter definition
class WardLinkage(AbstractLinkage):
    def alpha_fn(self, ni, nj, nk):
        return (ni + nk) / (ni + nj + nk)

    def beta_fn(self, ni, nj, nk):
        return (-nk) / (ni + nj + nk)

    def gamma_fn(self, ni, nj, nk):
        return 0


LINKAGES = {'single': SingleLinkage(),
            'complete': CompleteLinkage(),
            'ward': WardLinkage()}

'''
    Dendrogram node implementation. A node is a merge of its children. It also stores the distance between its children. 
'''
class DendrogramNode():
    def __init__(self, id, children=None):
        self.id = id;
        self.children = children;
        self.distance = 0;

    #Returns the list of the original datapoints contained in this node.
    def leaves(self):

        if (self.children):
            leaves = [];
            for child in self.children:
                leaves.extend(child.leaves())
            return leaves;
        else:
            return [self];

    #Recursive function that returns the children ids of this node along with their distance and number of datapoints
    #below it.
    def adjacency_list(self):

        if (self.children):
            al = [(self.id, self.children[0].id, self.children[1].id, self.distance, len(self))]
            for child in self.children:
                al.extend(child.adjacency_list());
            return al;
        else:
            return [];

    #Number of datapoints below this node.
    def __len__(self):
        return len(self.leaves())

'''
    Implements the dendrogram for hierarchical clustering.
'''
class Dendrogram(list):
    def __init__(self, items):
        super().__init__(map(DendrogramNode, items));
        self.n_clusters = len(items);

    def merge(self, *indexes):
        node = DendrogramNode(self.n_clusters, [self[i] for i in indexes]);
        self.n_clusters += 1;

        self[indexes[0]] = node;

        for i in indexes[1:]:
            del self[i];

    def to_adjacency_matrix(self):
        Z = self[0].adjacency_list();
        Z.sort()
        return np.array(Z)[:, 1:];

    def draw(self, title=None):
        fig = plt.figure();
        ax = plt.gca();

        A = self.to_adjacency_matrix();

        scipy_dendrogram(A, color_threshold=0.7 * max(A[:, 2]));

        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.suptitle(title, fontsize=12);
        plt.show();




'''
    Stores the distance matrix with distances between clusters.
'''
class DistanceMatrix():
    def __init__(self, X, linkage=SingleLinkage(), distance_metric=euclidean):
        self.linkage = linkage;
        self.distance_metric = distance_metric;
        self.dist_matrix = self.build_distance_matrix(X)
        self.dendrogram = Dendrogram([i for i in range(len(self.dist_matrix))]);

    #Looks for the smallest distance between clusters and returns the corresponding cluster indexes
    def get_smallest_distance(self):
        m = len(self.dist_matrix)
        i, j = np.unravel_index(np.argmin(self.dist_matrix), (m, m));
        return self.dist_matrix[i, j], i, j

    #Merges two clusters and updating the distance matrix and the dendrogram.
    def merge_clusters(self, i, j, distance):
        #print(self.linkage.__class__.__name__)
        self.dist_matrix = self.linkage.link(self.dist_matrix, self.dendrogram, i, j);
        self.dendrogram.merge(i, j)
        self.dendrogram[i].distance = distance;

    #Computes the distance matrix using the datapoints in X.
    def build_distance_matrix(self, X):
        m = len(X);
        dist_matrix = np.zeros((m, m));
        for i, j in combinations(range(m), 2):
            dist_matrix[i, j] = self.distance_metric(X[i], X[j]);
            dist_matrix[j, i] = dist_matrix[i, j]
        #Fill in the diagonal with np.inf so to eliminate their zero distances.
        np.fill_diagonal(dist_matrix, np.inf);

        return dist_matrix;

    def __len__(self):
        return len(self.dist_matrix);

'''
    Agglomemative clustering implementation.
'''
class AgglomerativeClustering():
    def __init__(self, linkage='single', distance_metric=euclidean, distance_thres=1.2):
        '''
        :param linkage: Clustering linkage: single, complete, ward
        :param distance_metric: distance metric to measure distance between clusters
        :param distance_thres: Threshold to stop clustering and return the clusters found.
        '''
        self.linkage = LINKAGES[linkage];
        self.distance_metric = distance_metric;
        self.distance_thres = distance_thres;

    def fit(self, X):
        m = len(X)
        self.dist_matrix = DistanceMatrix(X, linkage=self.linkage);
        dendrogram_thres = None;
        #While there are more than 1 cluster left.
        while (len(self.dist_matrix) > 1):
            distance, i, j = self.dist_matrix.get_smallest_distance();
            if (distance > self.distance_thres and dendrogram_thres is None):
                #When the threshold is reached, save the dendogram into memory for posterior visualization.
                dendrogram_thres = copy.copy(self.dist_matrix.dendrogram);

            self.dist_matrix.merge_clusters(i, j, distance);

        clusters = [];

        #Transform the dendrogram into a list of clusters.
        for node in dendrogram_thres:
            cluster = [node.id for node in node.leaves()]
            clusters.append(cluster);

        self.labels_ = np.full(m, -1);
        for cluster_id, cluster in enumerate(clusters):
            for x_ind in cluster:
                self.labels_[x_ind] = cluster_id;

    def dendrogram(self):
        return self.dist_matrix.dendrogram;
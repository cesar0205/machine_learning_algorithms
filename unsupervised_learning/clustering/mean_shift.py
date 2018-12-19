import numpy as np
from sklearn.neighbors import BallTree
from collections import defaultdict

#Calculates the new centroid of the window using a gaussian kernel to measure the distance of the current centroid
#to its neighbours and making a weighted sum. .
def exponential_kernel(x, points, bandwidth):
    diff = np.linalg.norm(x - points, axis=1, keepdims=True);
    weights = np.exp(-(diff / bandwidth) ** 2);
    return np.sum(points * weights, axis=0) / np.sum(weights);

#Calculates the new centroid of the window using the mean of all points within the window..
def lineal_kernel(x, points, bandwidth):
    return points.mean(axis=0);

'''
    Implements Mean Shift clustering. Looks for dense areas of datapoints and tries to cluster them. These dense areas
    are sliding windows that move their position towards areas where they could contain even more points.
    Finally we filter the sliding windows and remove the ones that are duplicates (their centroids are close to each 
    other)
'''
class MeanShift():
    def __init__(self, max_iterations=2000,
                 tol=1e-3,
                 update_kernel_fn=exponential_kernel,
                 min_bin_size=5,
                 bandwidth=2.5,
                 bin_length=0.1):
        self.max_iterations = max_iterations;
        self.update_kernel_fn = update_kernel_fn;
        self.bandwidth = bandwidth;
        self.tol = tol;
        self.min_bin_size = min_bin_size;
        self.bin_length = bin_length;


    def init_seeds(self, X):
        '''
        Creates a grid where the points will be binned. If a bin contains at least min_bin_size points, then it will be
        part of our initial centroid list (initial seeds).
        :param X: Dataset
        :return: Initial centroid list.
        '''
        grid = defaultdict(int);
        for x in X:
            binned_point = (x / self.bin_length).astype(int);
            grid[tuple(binned_point)] += 1;
        points = np.array([point for point, size in grid.items() if size >= self.min_bin_size]);
        return points * self.bin_length;

    def fit(self, X):
        centroids_dict = defaultdict(int);
        seeds = self.init_seeds(X);
        ball_tree = BallTree(X);
        for weighted_mean in seeds:
            for i in range(self.max_iterations):
                prev_weighted_mean = weighted_mean;
                points_within = X[ball_tree.query_radius([prev_weighted_mean], self.bandwidth)[0]];
                weighted_mean = self.update_kernel_fn(prev_weighted_mean, points_within, self.bandwidth);

                if (np.linalg.norm(weighted_mean - prev_weighted_mean) < self.tol * self.bandwidth):
                    break;

            centroids_dict[tuple(weighted_mean)] = len(points_within)

        self.centroids_ = self._remove_overlapping_windows(centroids_dict);
        self.labels_ = np.array([self._closest_centroid(x) for x in X])

        return self;

    def _remove_overlapping_windows(self, centroids_dict):
        '''
        Removes overlapping windows
        :param centroids_dict: Dictionary with windows positions and a list of points that each window has.
        :return: Filtered windows.
        '''
        centroids_by_intensity = sorted(centroids_dict.items(), key=lambda tup: tup[1], reverse=True);
        centroids = np.array([centroid for centroid, size in centroids_by_intensity if size >= self.min_bin_size]);
        unique = np.ones(len(centroids), dtype=bool)
        nbrs = BallTree(centroids);

        for centroid_ind, centroid in enumerate(centroids):
            if (unique[centroid_ind]):
                indexes = nbrs.query_radius([centroid], self.bandwidth)[0];
                unique[indexes] = False;
                unique[centroid_ind] = True;

        return centroids[unique];

    def _closest_centroid(self, x):
        return np.argmin(np.linalg.norm(x - self.centroids_, axis=1));



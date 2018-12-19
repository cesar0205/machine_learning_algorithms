import sklearn.datasets as dts
from itertools import cycle, islice
from scipy.sparse.linalg import ArpackNoConvergence
import matplotlib.pyplot as plt
import numpy as np
from unsupervised_learning.clustering.spectral_clustering import SpectralClustering
from unsupervised_learning.clustering.affinities import BasicAffinity, KnnAffinity, LocalScaling


def main():

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 500


    noisy_circles = dts.make_circles(n_samples=n_samples, factor=.5,noise=.05)
    noisy_moons = dts.make_moons(n_samples=n_samples, noise=.05)

    # Anisotropicly distributed data
    random_state = 170
    X, y = dts.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = dts.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    # Set up cluster parameters

    plt.figure(figsize=(21, 12.5))

    plot_num = 1

    datasets = [
        ("noisy circles", noisy_circles, {'n_clusters': 2}),
        ("noisy moons", noisy_moons, {'n_clusters': 2}),
        ("varied", varied, {'n_clusters': 3}),
        ("aniso", aniso, {'n_clusters': 3})]

    for i_dataset, (dataset_name, dataset, algo_params) in enumerate(datasets):

        X, y = dataset

        basic_sc = SpectralClustering(n_clusters=algo_params['n_clusters'], affinity = BasicAffinity())

        knn_sc = SpectralClustering(n_clusters=algo_params['n_clusters'], affinity = KnnAffinity())

        local_scaling_sc = SpectralClustering(n_clusters=algo_params['n_clusters'], affinity = LocalScaling())


        clustering_algorithms = (
            ('Basic affinity', basic_sc),
            ('Knn affinity', knn_sc),
            ('Local Scaling affinity', local_scaling_sc),
        )

        for affinity_name, algorithm in clustering_algorithms:
            print("Fitting model with dataset", dataset_name, "and affinity", affinity_name)
            error = False;
            try:
                algorithm.fit(X)
            except ArpackNoConvergence:
                #In case of error, then assign all datapoints to 0 label.
                error = True;
            if(error):
                y_pred = np.zeros((len(X), ), dtype=np.int)
            else:
                y_pred = algorithm.labels_.astype(np.int)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(affinity_name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1

    plt.show()

if __name__ == "__main__":
    main();
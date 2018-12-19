import matplotlib.pyplot as plt
from unsupervised_learning.clustering.gaussian_mixture import GaussianMixture
from sklearn.datasets import make_blobs


def main():

    X, y = make_blobs(n_samples=100, cluster_std=2, random_state=42)
    model = GaussianMixture();
    model.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
    plt.title("Mixture of gaussians")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    main();
from supervised_learning.svm.svm_primal import SVMPrimal
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from misc.plot_functions import plot_decision
import numpy as np

def main():
    # Create 4 centers so the algorithm tries to classify them correctly
    centers = np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]]);
    X, y = make_blobs(n_samples=100, centers=centers, cluster_std=0.5, random_state=42);
    y[y == 0] = -1;
    y[y == 1] = -1;
    y[y == 2] = 1;
    y[y == 3] = 1;
    plt.scatter(X[:, 0], X[:, 1], c=y)

    model = SVMPrimal(C=10);
    X = X.astype(float);
    y = y.astype(float);
    model.fit(X, y)


    plot_decision(model, X, y, "SVM Primal", "X0", "X1")

if __name__ == "__main__":
    main();
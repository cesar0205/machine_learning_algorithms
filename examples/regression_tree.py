from supervised_learning.trees.regression_tree import RegressionTree
import numpy as np
import matplotlib.pyplot as plt




def main():
    #100 samples of the sin(x) function in the range 0 - 2*pi in order to apply a regressor.
    T = 100;
    X = np.linspace(0, 2 * np.pi, T);
    y = np.sin(X);

    N = 30 #70 samples will be for training and 30 for testing.
    test_ind = np.random.choice(T, N, replace=False);

    train_ind = np.array([i for i in range(T) if i not in test_ind])

    X_test = X[test_ind].reshape(-1, 1)
    X_train = X[train_ind].reshape(-1, 1)
    y_train = y[train_ind]
    y_test = y[test_ind]

    model = RegressionTree()
    model.fit(X_train, y_train)
    print("Test score: ", model.score(X_test, y_test))
    plt.plot(X, y)
    pred = model.predict(X.reshape(T, 1))
    plt.plot(X, pred, 'o')
    plt.show()


if __name__ == "__main__":
    main();
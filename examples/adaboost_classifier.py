from sklearn.model_selection import train_test_split;
from sklearn.datasets import make_moons
from supervised_learning.trees.adaboost import Adaboost
from misc.plot_functions import plot_decision


def main():
    #We sample 100 points from the make_moons function in order to classify them.
    X, y = make_moons(n_samples=100, noise=0.25, random_state=42);
    #Adaboost works with signed targets.
    y[y == 0] = -1;
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = Adaboost()
    model.fit(X_train, y_train)
    print("Test score: ", model.score(X_test, y_test))
    plot_decision(model, X, y, "Adaboost classifier", "X0", "X1")


if __name__ == "__main__":
    main();
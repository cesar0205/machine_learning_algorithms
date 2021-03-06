import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap



def plot_decision(model, X, y, title, xlabel, ylabel):
    cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])
    x_s = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2);
    y_s = np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2);
    xx, yy = np.meshgrid(x_s, y_s);
    X_s = np.c_[xx.ravel(), yy.ravel()];
    zz = model.predict(X_s).reshape(xx.shape);
    plt.contourf(xx, yy, zz, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
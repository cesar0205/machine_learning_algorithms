import numpy as np

def accuracy_score(y, y_pred):
    return np.mean(y == y_pred);

def r2_score(y, y_pred):
    d1 = y - y_pred;
    d2 = y - y.mean();
    return 1 - d1.dot(d1)/d2.dot(d2);
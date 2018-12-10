import numpy as np

#one hot encoder
def to_categorical(y):
    labels = np.unique(y);
    n = len(labels);
    m = len(y);
    y_cat = np.zeros((m, n))
    y_cat[range(m), y] = 1;
    return y_cat;
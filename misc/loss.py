import numpy as np

class SquaredLoss():
    def loss(self, y, y_pred):
        return 0.5 * (y - y_pred) ** 2;

    def gradient(self, y, y_pred):
        return -(y - y_pred);

    def hessian(self, y, y_pred):
        return np.ones(y.shape);


class CrossEntropyLoss():
    def loss(self, y, y_pred):
        return - y * np.log2(y_pred) - (1 - y) * np.log2(1 - y_pred);

    def gradient(self, y, y_pred):
        return - y / y_pred + (1 - y) / (1 - y_pred)

    def hessian(self, y, y_pred):
        return y / np.power(y_pred, 2) + (1 - y) / np.power(1 - y_pred, 2)
import numpy as np
'''
    Exponential kernel to measure the similarity betweeen two points.
'''
class GaussianKernel():
    def __init__(self, sig1=0.01, sig2=0.1):
        self.sig1 = sig1;
        self.sig2 = sig2;

    def dot(self, x1, x2, sig1=None, sig2=None):
        if (sig1 is None):
            sig1 = self.sig1;
        if (sig2 is None):
            sig2 = self.sig2;

        diff = np.linalg.norm(x1 - x2);
        return np.exp(-(diff ** 2) / (2 * sig1 * sig2));
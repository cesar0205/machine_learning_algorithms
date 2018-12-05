import numpy as np
import cvxopt.solvers

class LinearKernel():
    def __init__(self):
        pass;

    def dot(self, x1, x2):
        return x1.dot(x2);

#Radial Basis Function
class RBFKernel():
    def __init__(self, gamma):
        self.gamma = gamma;

    def dot(self, x1, x2):
        diff = x1 - x2;
        return np.exp(-self.gamma * diff.dot(diff));

'''
    Class that models data using a SVM machine. The optmization problem is solved in the dual space to make use of kernels.
'''
class SVMDual():
    def __init__(self, C, kernel):
        self.C = C;
        self.kernel = kernel;

    def fit(self, X, y):

        # Minimize 0.5*(x.T).dot(P).dot(x) + q.T*x using cvxopt library
        #In particular solve the next problem:
        # 0.5*sum(sum(alpha_i*alpha_j*y_i*y_j*x_i*x_j)) - sum(alpha_i)
        #With the constraints:
        #alpha_i>0
        #sum(alpha_i*y_i) = 0
        #0< alpha_i < C

        m, n = X.shape;
        self.X = X;
        K = np.array([self.kernel.dot(X[i], X[j]) for j in range(m) for i in range(m)]).reshape(m, m);
        P = cvxopt.matrix(np.outer(y, y) * K);
        q = cvxopt.matrix(-np.ones(m))
        if (self.C == None):
            G = cvxopt.matrix(-np.identity(m));
            h = cvxopt.matrix(np.zeros(m));
        else:
            G1 = -np.identity(m);
            G2 = np.identity(m);
            G = cvxopt.matrix(np.vstack((G1, G2)))
            h1 = np.zeros(m);
            h2 = np.ones(m) * self.C;
            h = cvxopt.matrix(np.concatenate((h1, h2)))

        A = cvxopt.matrix(y, (1, m));
        b = cvxopt.matrix(0.0);

        result = cvxopt.solvers.qp(P, q, G, h, A, b);
        alphas = np.ravel(result['x']);
        mask = alphas > 1e-7;
        self.sv_a = alphas[mask];
        self.sv_X = X[mask];
        self.sv_y = y[mask];

        # b_i = y - w*x_i for a given sample
        # Average over all b_i's

        self.b = np.mean([self.sv_y[i] - self.wDot(self.sv_X[i]) for i in range(len(self.sv_X))]);

        if (isinstance(self.kernel, LinearKernel)):
            self.w = np.sum([self.sv_a[i] * self.sv_y[i] * self.sv_X[i] for i in range(len(self.sv_X))], axis=0);

    def wDot(self, x):
        return np.sum([self.sv_a[i] * self.sv_y[i] * self.kernel.dot(self.sv_X[i], x) for i in range(len(self.sv_X))]);

    def predict(self, X):
        if (isinstance(self.kernel, LinearKernel)):
            return np.sign(X.dot(self.w) + self.b);
        else:
            return np.array([np.sign(self.wDot(x) + self.b) for x in X])
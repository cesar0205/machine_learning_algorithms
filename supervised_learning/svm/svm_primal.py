import numpy as np
import cvxopt.solvers

'''
    Class that models data using a SVM machine. The optmization problem is solved in the primal space. We cannot use non linear kernels.
'''
class SVMPrimal():
    def __init__(self, C):
        self.C = C;

    def fit(self, X, y):

        # Minimize 0.5*(x.T).dot(P).dot(x) + q.T*x using cvxopt library
        #In particular solve the next problem:
        # 0.5*||w||^2 + C*sum(epsilon_i)
        #With the constraints:
        #y_i*(w*x_i + b) > 1 - epsilon_i
        #epsilon_i > 0
        X_new = np.c_[np.ones(len(X)), X];
        m, n = X_new.shape;
        P = np.identity(m + n);
        P[n:, n:] = 0;
        P[0, 0] = 0;
        P = cvxopt.matrix(P);

        q = cvxopt.matrix(np.concatenate((np.zeros(n), np.ones(m) * self.C)))

        G1 = -y.reshape(m, 1) * X_new;
        G2 = -np.identity(m)
        G3 = np.zeros((m, n))
        G4 = -np.identity(m)

        G12 = np.c_[G1, G2];
        G34 = np.c_[G3, G4];
        G = cvxopt.matrix(np.vstack((G12, G34)));

        h = cvxopt.matrix(np.concatenate((-np.ones(m), np.zeros(m))));

        result = cvxopt.solvers.qp(P, q, G, h);
        self.w = result['x'][:n]

    def predict(self, X):
        X_new = np.c_[np.ones(len(X)), X];
        return np.sign(X_new.dot(self.w))

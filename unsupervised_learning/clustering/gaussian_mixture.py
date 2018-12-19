from scipy.stats import multivariate_normal
import numpy as np

'''
    Mixture of gaussians for clustering (Expectation-Maximization algorithm).
'''
class GaussianMixture:
    def __init__(self, n_clusters=3, max_iterations=2000, tol=1e-7):
        self.n_clusters = n_clusters;
        self.max_iterations = max_iterations;
        self.tol = tol;

    def init_centroids(self, X):
        m = len(X);
        self.priors = np.ones(self.n_clusters) / self.n_clusters;
        for i in range(self.n_clusters):
            param = {};
            param['mean'] = X[np.random.choice(m)];
            mean = X.mean(axis=0);
            param['cov'] = (1 / (m - 1)) * (X - mean).T.dot(X - mean);
            self.parameters[i] = param

    def get_likelihood(self, X):
        m = len(X);
        likelihood = np.zeros((m, self.n_clusters));
        for i in range(self.n_clusters):
            mean = self.parameters[i]['mean'];
            cov = self.parameters[i]['cov'];
            likelihood[:, i] = multivariate_normal.pdf(X, mean, cov)
        return likelihood;

    #Expectation step.
    def expectation(self, X):
        self.prev_pz_x = self.pz_x;
        #p(x, z)
        pxz = self.get_likelihood(X) * self.priors;
        #p(z| x)
        self.pz_x = pxz / np.sum(pxz, axis=1, keepdims=True);
        self.assigments = np.argmax(self.pz_x, axis=1);

    #Expectation step
    def maximization(self, X):
        m = len(X);
        for i in range(self.n_clusters):
            #Get p(z| x) for cluster i and calculate new mean and cov for cluster i
            pz_x = np.expand_dims(self.pz_x[:, i], axis=1);
            mean = np.sum(pz_x * X, axis=0) / np.sum(pz_x);
            cov = (pz_x * (X - mean)).T.dot(X - mean) / np.sum(pz_x);
            self.parameters[i]['mean'] = mean;
            self.parameters[i]['cov'] = cov;
            self.priors[i] = np.sum(pz_x) / m;

    #Train the mixture of gaussians using the EM algorithm.
    def fit(self, X):
        m = len(X);
        self.parameters = [None] * self.n_clusters;
        self.prev_pz_x = None;
        self.pz_x = np.full((m, self.n_clusters), np.inf)

        self.init_centroids(X);
        for i in range(self.max_iterations):
            self.expectation(X);
            self.maximization(X);
            if (np.linalg.norm(self.pz_x - self.prev_pz_x) < self.tol):
                break;

        self.expectation(X)
        self.labels_ = self.assigments;

        return self;

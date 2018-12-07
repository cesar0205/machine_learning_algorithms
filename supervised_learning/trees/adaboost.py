import numpy as np
from misc.metrics import accuracy_score

class DecisionStump():
    def __init__(self, col=None, thres=None, pol=None, alpha=None):
        self.col = col;
        self.thres = thres;
        self.pol = pol;
        self.alpha = alpha;


class Adaboost():
    '''
        Adaboost implementation using a decision stump on each tree. That is, we use a tree of depth 1.
    '''
    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators;

    def fit(self, X, y):
        m, n = X.shape
        w = np.ones(m) / m;
        self.models = [];
        for i in range(self.n_estimators):
            #Create a decision strum on each tree.
            dt = DecisionStump();
            min_error = np.inf;
            cols = range(n);
            #Try splitting the data on each attribute.
            for col in cols:
                thresholds = np.unique(X[:, col]);
                for thres in thresholds:
                    pol = 1;
                    prediction = np.ones(m);
                    prediction[X[:, col] < thres] = -1;
                    error = np.sum(w[prediction != y])

                    #If the error is greater than 0.5, the classification sign is changed.
                    if (error > 0.5):
                        pol = -1;
                        error = 1 - error;

                    #Save the attribute and its value that minimizes the error.
                    if (error < min_error):
                        min_error = error;
                        dt.col = col;
                        dt.thres = thres;
                        dt.pol = pol;

            #Update the alpha and predictions according the adaboost algorithm.
            dt.alpha = 0.5 * (np.log(1 - min_error) - np.log(min_error));
            prediction = np.ones(m);
            neg_ind = dt.pol * X[:, dt.col] < dt.pol * dt.thres;
            prediction[neg_ind] = -1;

            #Update and normalize the weights according to the adaboost algorithm.
            w *= np.exp(-dt.alpha * prediction * y);
            w /= np.sum(w)
            self.models.append(dt);

        return self;

    def predict(self, X):
        m = len(X)
        final_pred = np.zeros(m);
        for dt in self.models:
            prediction = np.ones(m);
            neg_ind = dt.pol * X[:, dt.col] < dt.pol * dt.thres;
            prediction[neg_ind] = -1;
            #The predictions made by each decision stump is weighted by its alpha parameter.
            final_pred += dt.alpha * prediction;

        return np.sign(final_pred)

    def score(self, X, y):
        y_pred = self.predict(X);
        return accuracy_score(y, y_pred)
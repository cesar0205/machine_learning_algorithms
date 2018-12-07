from supervised_learning.trees.basic_trees import ClassificationTree, RegressionTree
from misc.metrics import accuracy_score, r2_score
import numpy as np


class BaggedClassificationTree():
    '''
        Bagged tree classifier.
        Ensemble of tree classifiers where each tree is fitted with a bagged subset of the training data.
    '''
    def __init__(self, n_estimators=200,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2,
                 random=False,
                 max_features=None):
        self.n_estimators = n_estimators;
        self.max_depth = max_depth;
        self.min_impurity = min_impurity;
        self.min_samples_split = min_samples_split;
        self.random = random;
        self.max_features = max_features;

        self.models = [];
        for i in range(self.n_estimators):
            dt = ClassificationTree(max_depth=self.max_depth,
                                    min_impurity=self.min_impurity,
                                    min_samples_split=self.min_samples_split,
                                    random=self.random,
                                    max_features=self.max_features);
            self.models.append(dt);

    def fit(self, X, y):
        m = len(X);
        self.classes_ = np.unique(y);
        for dt in self.models:
            rand_ind = np.random.choice(m, m, replace=True);
            Xb = X[rand_ind];
            yb = y[rand_ind];
            dt.fit(Xb, yb);

    def predict(self, X):
        m = len(X);
        final_pred = np.zeros((m, len(self.classes_)));
        for dt in self.models:
            final_pred[range(m), dt.predict(X)] += 1;
        return np.argmax(final_pred, axis=1);

    def score(self, X, y):
        y_pred = self.predict(X);
        return accuracy_score(y, y_pred);


class BaggedRegressionTree():
    '''
        Bagged tree regressor.
        Ensemble of tree regressors where each tree is fitted with a bagged subset of the training data.
    '''
    def __init__(self, n_estimators=200,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2,
                 random=False,
                 max_features=None):
        self.n_estimators = n_estimators;
        self.max_depth = max_depth;
        self.min_impurity = min_impurity;
        self.min_samples_split = min_samples_split;
        self.random = random;
        self.max_features = max_features;

        self.models = [];
        for i in range(self.n_estimators):
            dt = RegressionTree(max_depth=self.max_depth,
                                min_impurity=self.min_impurity,
                                min_samples_split=self.min_samples_split,
                                random=self.random,
                                max_features=self.max_features);
            self.models.append(dt);

    def fit(self, X, y):
        m = len(X);
        self.classes_ = np.unique(y);
        for dt in self.models:
            rand_ind = np.random.choice(m, m, replace=True);
            Xb = X[rand_ind];
            yb = y[rand_ind];
            dt.fit(Xb, yb);

    def predict(self, X):
        m = len(X);
        final_pred = np.zeros(m);
        for dt in self.models:
            final_pred += dt.predict(X).flatten();
        return final_pred / self.n_estimators

    def score(self, X, y):
        y_pred = self.predict(X);
        return r2_score(y, y_pred);
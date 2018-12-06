from misc.metrics import r2_score
from supervised_learning.trees.decision_tree import DecisionTree
import numpy as np

class RegressionTree(DecisionTree):
    '''Regression Tree regressor
        Extends from DecisionTree.
    '''
    def _calculate_leaf_value(self, y):
        #The leaf value is calculated with the mean of the samples.
        return y.mean(0);

    def _impurity(self, X):
        #The impurity is calculated with the variance of the data.
        m = len(X);
        X_mean = X.mean(0);

        return np.diag((1 / m) * (X - X_mean).T.dot(X - X_mean))

    def score(self, X, y):
        #For regression we use the r2 score
        y_pred = self.predict(X).flatten();
        return r2_score(y, y_pred);
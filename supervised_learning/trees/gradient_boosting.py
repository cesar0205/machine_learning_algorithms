import numpy as np
from supervised_learning.trees.basic_trees import RegressionTree
from supervised_learning.trees.decision_tree import DecisionTree
from misc.loss import SquaredLoss, CrossEntropyLoss
from misc.metrics import accuracy_score, r2_score

#https://xgboost.readthedocs.io/en/latest/tutorials/model.html

class XGBoostRegressionTree(DecisionTree):

    #Split y into its gradient matrix and its hessian matrix.
    def _split(self, y):
        n = y.shape[1] // 2;
        return y[:, :n], y[:, n:];

    #Score on a given leaf
    def _score(self, grad, hess):
        numerator = np.sum(np.power(np.sum(grad, axis=0), 2))
        denominator = np.sum(hess);
        return numerator / denominator;

    #The leaf value is calculated with the mean of the gradient
    def _calculate_leaf_value(self, y):
        gradient, hessian = self._split(y);
        return gradient.mean(0);

    #The calculation is almost identical to the one of the DecisionTree, however in this case it is computed
    #using the definition of score for XGBoost
    def _impurity_reduction(self, x, y, thres):
        m = len(x)
        left_ind, right_ind = self._split_ind(x, thres);

        if (len(left_ind) == 0 or len(left_ind) == m):
            return 0.0;

        y_left = y[left_ind];
        y_right = y[right_ind];

        left_grad, left_hess = self._split(y_left);
        right_grad, right_hess = self._split(y_right);
        total_grad, total_hess = self._split(y);

        left_score = self._score(left_grad, left_hess);
        right_score = self._score(right_grad, right_hess);
        total_score = self._score(total_grad, total_hess);

        return 0.5 * (left_score + right_score - total_score)


#http://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/slides/gradient_boosting.pdf

class GradientBoosting():
    '''
        Implements gradient boosting in two flavors: The first one using an ensemble of simple regression trees and
        the second one using an ensemble of XGBoosted regression trees, which construct what is known as XGBoosted
        trees.

        The main difference between them is how the score of a leaf is computed for the impurity reduction calculation.
        For regression trees, the score of a leaf is the samples variance. For XGBoosted regression trees, the score
        of a leaf is GL^2/(HL + lambda).
    '''
    def __init__(self, n_estimators=200,
                 learning_rate=0.5,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2,
                 loss=SquaredLoss(),
                 xgboost=False):

        self.n_estimators = n_estimators;
        self.learning_rate = learning_rate;
        self.xgboost = xgboost;
        self.loss = loss;

        self.regression_tree = RegressionTree;
        self.get_gradient = self.gradient;

        if (self.xgboost):
            self.regression_tree = XGBoostRegressionTree;
            self.get_gradient = self.gradient_hessian;

        self.models = []
        for i in range(self.n_estimators):
            dt = self.regression_tree(max_depth=max_depth,
                                      min_impurity=min_impurity,
                                      min_samples_split=min_samples_split);
            self.models.append(dt);

    def gradient(self, y, y_pred):
        return self.loss.gradient(y, y_pred)

    def gradient_hessian(self, y, y_pred):
        gradient = self.loss.gradient(y, y_pred);
        hessian = self.loss.hessian(y, y_pred);
        return np.c_[gradient, hessian];

    def fit(self, X, y):
        if (len(y.shape) == 1):
            y = np.expand_dims(y, axis=1);
        y_pred = np.full(y.shape, y.mean(0));
        self.n_classes_ = y_pred.shape[1];
        for dt in self.models:
            gradient = self.get_gradient(y, y_pred);
            dt.fit(X, gradient);
            update = dt.predict(X);
            # print(self.learning_rate)
            y_pred -= np.multiply(self.learning_rate, update)

        return self;

    def predict(self, X):
        m = len(X);
        y_pred = np.zeros((m, self.n_classes_));
        for dt in self.models:
            update = dt.predict(X);
            y_pred -= np.multiply(self.learning_rate, update)

        return y_pred;



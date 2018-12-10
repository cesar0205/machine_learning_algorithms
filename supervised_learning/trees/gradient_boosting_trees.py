import numpy as np
from supervised_learning.trees.gradient_boosting import GradientBoosting
from misc.loss import SquaredLoss, CrossEntropyLoss
from misc.metrics import accuracy_score, r2_score
from misc.auxiliar_functions import to_categorical


class GradientBoostingClassifier(GradientBoosting):
    '''
        Gradient Boosting for classification
    '''
    def __init__(self, n_estimators=200,
                 learning_rate=0.5,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2,
                 xgboost=False):
        super().__init__(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         max_depth=max_depth,
                         min_impurity=min_impurity,
                         min_samples_split=min_samples_split,
                         loss=CrossEntropyLoss(),
                         xgboost=xgboost);

    def fit(self, X, y):
        y_cat = to_categorical(y);
        super().fit(X, y_cat);
        return self;

    def predict(self, X):
        y_pred = super().predict(X);
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True);
        return np.argmax(y_pred, axis=1);

    def score(self, X, y):
        y_pred = self.predict(X);
        return accuracy_score(y, y_pred);


class GradientBoostingRegressor(GradientBoosting):
    '''
        Gradient Boosting for regression
    '''
    def __init__(self, n_estimators=200,
                 learning_rate=0.5,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2,
                 xgboost=False):
        super().__init__(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         max_depth=max_depth,
                         min_impurity=min_impurity,
                         min_samples_split=min_samples_split,
                         loss=SquaredLoss(),
                         xgboost=xgboost);

    def predict(self, X):
        return super().predict(X).flatten();

    def score(self, X, y):
        y_pred = self.predict(X);
        return r2_score(y, y_pred);
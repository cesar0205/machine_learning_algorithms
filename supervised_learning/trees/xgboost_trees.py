import numpy as np
from supervised_learning.trees.gradient_boosting_trees import GradientBoostingClassifier, GradientBoostingRegressor

class XGBoostClassifier(GradientBoostingClassifier):
    '''
        XGBoost for classification
    '''
    def __init__(self, n_estimators=200,
                 learning_rate=0.5,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2):
        super().__init__(n_estimators=200,
                         learning_rate=learning_rate,
                         max_depth=max_depth,
                         min_impurity=min_impurity,
                         min_samples_split=min_samples_split,
                         xgboost=True);


class XGBoostRegressor(GradientBoostingRegressor):
    '''
        XGBoost for regression
    '''
    def __init__(self, n_estimators=200,
                 learning_rate=0.5,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2):
        super().__init__(n_estimators=200,
                         learning_rate=learning_rate,
                         max_depth=max_depth,
                         min_impurity=min_impurity,
                         min_samples_split=min_samples_split,
                         xgboost=True);

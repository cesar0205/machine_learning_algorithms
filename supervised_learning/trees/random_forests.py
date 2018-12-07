from supervised_learning.trees.bagged_trees import BaggedRegressionTree, BaggedClassificationTree
import numpy as np

class RandomForestClassifier(BaggedClassificationTree):
    '''
    Random forest classifier.
    The feature subset is randomly selected at each node within each tree, not at each tree as proposed by Leo Breiman.
    '''
    def __init__(self, n_estimators=200,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2,
                 max_features = None):
        super().__init__(n_estimators,
                         max_depth,
                         min_impurity,
                         min_samples_split,
                         random=True,
                         max_features=max_features)


class RandomForestRegressor(BaggedRegressionTree):
    '''
    Random forest regressor.
    The feature subset is randomly selected at each node within each tree, not at each tree as proposed by Leo Breiman.
    '''
    def __init__(self, n_estimators=200,
                 max_depth=np.inf,
                 min_impurity=0,
                 min_samples_split=2,
                 max_features=None):
        super().__init__(n_estimators,
                         max_depth,
                         min_impurity,
                         min_samples_split,
                         random=True,
                         max_features=max_features)
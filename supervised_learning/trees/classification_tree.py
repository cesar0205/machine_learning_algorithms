from misc.metrics import accuracy_score
from supervised_learning.trees.decision_tree import DecisionTree
import numpy as np

class ClassificationTree(DecisionTree):
    '''
    ClassificationTree classifier.
    '''
    def _calculate_leaf_value(self, y):
        #The leaf value is the mode element in the data.
        labels = np.unique(y);
        max_label = None;
        max_count = 0
        for label in labels:
            count = len(y[y == label])
            if (count > max_count):
                max_count = count;
                max_label = label;
        return max_label;

    def _impurity(self, y):
        #The impourity is calculated with the entropy of the data.
        m = len(y);
        labels = np.unique(y);
        tot_entropy = 0;
        for label in labels:
            p = len(y[y == label]) / m;
            tot_entropy -= p * np.log2(p)

        return tot_entropy;

    def predict(self, X):
        return super().predict(X).flatten();

    def score(self, X, y):
        y_pred = self.predict(X);
        return accuracy_score(y, y_pred);
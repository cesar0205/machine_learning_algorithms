import numpy as np
'''
    Decision Tree node that points to subsequent children node. 
'''
class DecisionNode():
    def __init__(self, col=None, thres=None, left=None, right=None, prediction=None):
        self.col = col;
        self.thres = thres;
        self.left = left;
        self.right = right;
        self.prediction = prediction;

'''
    Base to construct Decision Tree classifier and regressor.
'''
class DecisionTree():
    '''
    max_depth: int
       Maximum depth of the tree
    min_impurity: float
        Minimum impurity of the tree
    min_samples_split: int
        Miminum samples in a node to consider a split
    random: boolean
        Whether to take a random subsample of the features during a split.
    max_features: int
        How many features consider during a split.
    '''
    def __init__(self, max_depth=np.inf, min_impurity=0, min_samples_split=2, random=False, max_features=None):
        self.max_depth = max_depth;
        self.min_impurity = min_impurity;
        self.min_samples_split = min_samples_split;
        self.random = random;
        self.max_features = max_features;

    def _calculate_leaf_value(self, y):
        raise NotImplementedError();

    def _impurity(self, y):
        raise NotImplementedError();

    def fit(self, X, y):
        if (len(y.shape) == 1):
            y = np.expand_dims(y, axis=1);
        self.root = self._build_tree(X, y, 0);
        return self;

    def _build_tree(self, X, y, depth):
        m, n = X.shape;
        if (depth <= self.max_depth and m >= self.min_samples_split):

            cols = range(n);
            max_ir = 0;
            best_col = 0;
            best_thres = 0;

            if (self.random):
                if (self.max_features is None):
                    self.max_features = int(np.sqrt(n));
                cols = np.random.choice(n, self.max_features, replace=False);

            for col in cols:
                ir, thres = self._find_split(X[:, col], y);
                if (ir > max_ir):
                    max_ir = ir;
                    best_col = col;
                    best_thres = thres;

            if (max_ir > self.min_impurity):
                left_ind, right_ind = self._split_ind(X[:, best_col], best_thres);

                left = self._build_tree(X[left_ind], y[left_ind], depth + 1);
                right = self._build_tree(X[right_ind], y[right_ind], depth + 1);

                return DecisionNode(best_col, best_thres, left, right, None);

        return DecisionNode(prediction=self._calculate_leaf_value(y));

    def _find_split(self, x, y):
        m = len(y)
        thresholds = np.unique(x);
        max_ir = 0;
        best_thres = None;
        for thres in thresholds:
            ir = self._impurity_reduction(x, y, thres);
            if (ir > max_ir):
                max_ir = ir;
                best_thres = thres;

        return max_ir, best_thres;

    def _split_ind(self, x, thres):

        if (isinstance(thres, int) or isinstance(thres, float)):
            divide = lambda sample: sample >= thres;
        else:
            divide = lambda sample: sample == thres;

        left_ind = np.array([i for i, sample in enumerate(x) if divide(sample)])
        right_ind = np.array([i for i, sample in enumerate(x) if not divide(sample)])

        return left_ind, right_ind;

    def _impurity_reduction(self, x, y, thres):
        m = len(y);
        left_ind, right_ind = self._split_ind(x, thres);

        if (len(left_ind) == 0 or len(left_ind) == m):
            return 0;

        y_left = y[left_ind];
        y_right = y[right_ind];

        p0 = len(y_left) / m;
        p1 = 1 - p0;

        ir = self._impurity(y) - p0 * self._impurity(y_left) - p1 * self._impurity(y_right);

        return np.sum(ir);

    def _predict_one(self, x, node=None):

        if (node is None):
            node = self.root;

        if (node.prediction is not None):
            return node.prediction;

        branch = node.right;
        attribute = x[node.col];
        if (isinstance(attribute, int) or isinstance(attribute, float)):
            if (attribute >= node.thres):
                branch = node.left;
        elif (attribute == node.thres):
            branch = node.left;

        return self._predict_one(x, branch);

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X]);
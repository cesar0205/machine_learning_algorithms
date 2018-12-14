import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier

'''
    Stacking example using a darts dataset. The objective is to predict which player threw a given dart.
'''

class TargetExtractor(BaseEstimator, TransformerMixin):
    '''
        Transformer that maps from competitor to id.
    '''
    def __init__(self):
        pass;

    def fit(self, X, y=None):
        return self;

    def transform(self, X, y=None):
        return X['Competitor'].map({'Bob': 0, 'Sue': 1, 'Kate': 2, 'Mark': 3}).values


class FeaturesTransformer(BaseEstimator, TransformerMixin):
    '''
        Transformer that creates a new feature 'radius' using the x and y coordinates
    '''
    def __init__(self):
        pass;

    def fit(self, X, y=None):
        return self;

    def transform(self, X, y=None):
        X['Radius'] = X.apply(lambda x: np.sqrt(x[1] ** 2 + x[2] ** 2), axis=1);
        return X[['XCoord', 'YCoord', 'Radius']].values

def main():

    #Stackinng does't have its own class as other ensembles in this module. This is because it requires
    #careful inspection of the base models to be sure that they different parts of the training data. So different
    #classifiers can be used as base models depending on the problem at hand.

    #Read train and test data.
    darts_train = pd.read_csv("../datasets/darts_train.csv")
    darts_test = pd.read_csv("../datasets/darts_test.csv")

    #Preprocess train and test data.
    trans = FeaturesTransformer();
    target = TargetExtractor();
    X_train = trans.fit_transform(darts_train)
    y_train = target.fit_transform(darts_train)
    X_test = trans.fit_transform(darts_test)
    y_test = target.fit_transform(darts_test)

    #Lets try first a linear SVC and use grid search to search for the best hyperparameter C
    svm_model = LinearSVC()

    param_grid = {'C':[0.01, 0.1, 1, 10, 100]}
    svm_grid = GridSearchCV(svm_model, param_grid, cv = 3)
    svm_grid.fit(X_train, y_train);
    svm_model = svm_grid.best_estimator_
    print("Test score for SVM model: ", svm_model.score(X_test, y_test))

    #Now lets try a KNN and use grid search to find the best hyperparameter n_neighbors.
    knn_model = KNeighborsClassifier()

    param_grid = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
    knn_grid = GridSearchCV(knn_model, param_grid, cv = 3)
    knn_grid.fit(X_train, y_train);
    knn_model = knn_grid.best_estimator_
    print("Test score for the KNN model: ", knn_model.score(X_test, y_test))


    #Compare the predictions made by each model. Be sure that they make errors in different regions. That's
    #where staking and in general using ensembles is useful.

    #Use this models as the base for our blender and create a new train data with the predictions
    #of each model.
    base_models = [svm_model, knn_model];
    n = len(base_models);
    m = len(X_train)
    base_predictions = np.zeros((m, n))
    n_folds = 5;
    splits = KFold(n_splits=n_folds);

    #To avoid overfitting create two datasets out of the training data: A new train set to fit each
    #model and a new test data set to create predictions from it.
    for train_ind, test_ind in splits.split(X_train, y_train):

        for model_ind, model in enumerate(base_models):
            model.fit(X_train[train_ind], y_train[train_ind]);
            predictions = model.predict(X_train[test_ind]);
            base_predictions[test_ind, model_ind] = predictions;

    #Now create a blender model that will be our final model. It is fitted with the base predictions
    #from the two base models.

    #Use grid search to find the best hyperparameters.

    blender = RandomForestClassifier()
    parameter_grid = {'n_estimators': [10, 50, 100, 150, 200]}
    blender_grid = GridSearchCV(blender, param_grid=parameter_grid, cv=3)
    blender_grid.fit(base_predictions, y_train)

    blender = blender_grid.best_estimator_

    #Now that the blender is fitted and sure it doesn't overfits the training data we can retrain the
    #base models using the whole training set as won't affect the blender.
    for model in base_models:
        model.fit(X_train, y_train);


    #Calculate the accuracy score on the test data.
    m = len(X_test);
    n = len(base_models);
    base_predictions = np.zeros((m, n), int);

    for model_ind, model in enumerate(base_models):
        base_predictions[:, model_ind] = model.predict(X_test)

    #Now we print the test score for the staked model.
    #Ideally the score will be equal or greater than any of the base models.
    print("Test score for the staked model:", blender.score(base_predictions, y_test))

if __name__ == "__main__":
    main();
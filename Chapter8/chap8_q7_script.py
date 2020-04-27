import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter8')


def main():
    ## Load dataframe
    boston_df = pd.read_csv(os.path.join(DATA_DIR, "boston.csv"))
    # print(boston_df.info())
    ## Set the predictor and target dataframes
    X = boston_df.drop('target', axis=1)
    y = boston_df['target']

    # print(X)
    # print(y)

    ## Instantiate the RandomForestRegressor
    f_model = RandomForestRegressor()

    ### For loop for cross validation test of number of predictors considered at each step
    ## Create an array of integers to iterate over
    num_feats = np.arange(1,X.shape[1])
    ## Set the number of trees to be constant.
    f_model.set_params(n_estimators=50)
    ## Create empty lists for training and testing scores
    test_cv_scores_feats = []
    train_cv_scores_feats = []
    ## Loop over the number of features
    for i in num_feats:
        print(i)
        ## Update the regressor and cross_validate
        f_model.set_params(max_features = i)
        scores = cross_validate(f_model, X, y, return_train_score=True)
        test_cv_scores_feats.append(scores['test_score'].mean())
        train_cv_scores_feats.append(scores['train_score'].mean())

    ### For loop for cross validation test of number of trees
    ## Create an array of integers to iterate over
    num_trees = np.arange(1,50,1)
    ## Set the number of predictors considered at each step to be constant. In this case we use the total number of predictors
    f_model.set_params(max_features = X.shape[1])
    ## Create empty lists for training and testing scores
    test_cv_scores_ntrees = []
    train_cv_scores_ntrees = []
    ## Loop over the number of trees
    for trees in num_trees:
        print(trees)
        f_model.set_params(n_estimators = trees)
        scores = cross_validate(f_model, X, y, return_train_score=True)
        test_cv_scores_ntrees.append(scores['test_score'].mean())
        train_cv_scores_ntrees.append(scores['train_score'].mean())

    ## Plot the results of each cv loop. Both training and testing.
    fig, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols=1, figsize=(12,12))
    ax1.plot(num_feats, test_cv_scores_feats, label='test score')
    ax1.plot(num_feats, train_cv_scores_feats, label='train_score')

    ax2.plot(num_trees, test_cv_scores_ntrees, label='test score')
    ax2.plot(num_trees, train_cv_scores_ntrees, label='train_score')

    ax1.set_xlabel('Number of features')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for number of features considered')
    ax1.legend(fontsize = 'large')

    ax2.set_xlabel('Number of Trees in forest')
    ax2.set_ylabel('Score')
    ax2.set_title('Model performance for number of trees considered')
    ax2.legend(fontsize = 'large')

    plt.savefig(os.path.join(IMAGE_DIR, 'q7_model_performance.png'))
    plt.show()

if __name__ == "__main__":
    main()

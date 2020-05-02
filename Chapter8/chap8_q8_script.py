import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter8')

np.random.seed(101)

def part_b(X,y):
    ## Split in test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    ## Instantiate the decision tree
    tree = DecisionTreeRegressor()
    ## FIt the tree
    tree.fit(X_train, y_train)
    ## Score the tree
    score = tree.score(X_test, y_test)
    print('Test score: ', round(score,2))
    score = tree.score(X_train, y_train)
    print('Train score: ', round(score,2))

    ## Plot the decision tree
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(36,36))
    # plot_tree(tree, ax=ax)
    # plt.show()

def part_c(X,y):
    ## Instantiate the decision tree
    tree = DecisionTreeRegressor()

    ## Create set of test alpha values and empty cv scores for training and testing
    alpha_value = np.arange(0,1, 0.001)
    cv_score_alpha_test = []
    cv_score_alpha_train = []

    ## iteracte of alpha and score the training and test results
    for a in alpha_value:
        print(a)
        tree.set_params(ccp_alpha = a)
        scores = cross_validate(tree, X, y, return_train_score=True)
        cv_score_alpha_test.append(scores['test_score'].mean())
        cv_score_alpha_train.append(scores['train_score'].mean())

    ## Plot the training and test scores
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(alpha_value, cv_score_alpha_test, label = 'test score')
    ax.plot(alpha_value, cv_score_alpha_train, label = 'training score')

    ax.set_xlabel(r'\alpha', fontsize = 'large')
    ax.set_ylabel('Score (R -squared)', fontsize = 'large')
    ax.set_title('Model performance as pruning is increased', fontsize = 'xx-large')
    ax.legend(fontsize = 'large')

    plt.savefig(os.path.join(IMAGE_DIR, 'q8_tree_complexity_plot.png'))
    plt.show()
    plt.close()

def part_d(X,y):
    ## Bagging approach
    ## Split in test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    ## Store a list of column names
    cols = X.columns
    ## Instaniate the regressor
    tree = DecisionTreeRegressor()
    ## Instantiate and fit the bagger
    bagger = BaggingRegressor(base_estimator=tree)
    bagger.fit(X_train,y_train)

    ## Score the test set with the bagger
    print('Test Score: ', bagger.score(X_test, y_test))

    ## Create a dictionary of feature importances in the modeal and sort them
    feat_import = bagger.estimators_[0].feature_importances_
    feat_import = dict(zip(cols, feat_import))
    feat_import = {k: v for k, v in sorted(feat_import.items(), key=lambda item: item[1], reverse = True)}
    for key, val in feat_import.items():
        print("{0:<20} : {1:<20}".format(key, round(val,3)))
    # print(bagger.estimators_[0].feature_importances_)

def part_e(X,y):
    ## Split in test and training sets
    cols = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    ## Instantiate the decision tree
    forest = RandomForestRegressor()
    ## Fit the tree
    forest.fit(X_train, y_train)
    ## Score the tree
    score_R2 = forest.score(X_test, y_test)
    print('Test score (R-squared): ', round(score_R2,2))

    preds = forest.predict(X_test)

    score_MSE = mean_squared_error(y_test, preds)
    print('Test score (MSE): ', round(score_MSE,2))

    feat_import = forest.feature_importances_
    feat_import = dict(zip(cols, feat_import))
    # print(feat_import)
    ## Sort by feature_importance
    feat_import = {k: v for k, v in sorted(feat_import.items(), key=lambda item: item[1], reverse=True)}
    for key, val in feat_import.items():
        print("{0:<20} : {1:<20}".format(key, round(val,3)))
    # print(feat_import)

def main():
    ## Load dataframe
    carseats_df = pd.read_csv(os.path.join(DATA_DIR, "carseats.csv"))
    carseats_df.set_index('Unnamed: 0', inplace=True)
    print(carseats_df.info())

    ##Preprocessing
    carseats_df = pd.get_dummies(carseats_df,['Urban', 'US', 'ShelveLoc'], drop_first=True)

    y = carseats_df['Sales']
    X = carseats_df.drop('Sales', axis=1)
    # print(X)
    # print(y)


    # part_b(X,y)
    # part_c(X,y)
    # part_d(X,y)
    part_e(X,y)




if __name__ == "__main__":
    main()

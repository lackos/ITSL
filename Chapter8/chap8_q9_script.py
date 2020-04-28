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
from sklearn.tree import DecisionTreeClassifier

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter8')

def part_b(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=800)
    ## Instantiate the decission tree classifier
    tree = DecisionTreeClassifier()

    tree.fit(X_train, y_train)
    test_score = tree.score(X_test,y_test)
    train_score = tree.score(X_train, y_train)
    depth = tree.get_depth()
    num_leaves= tree.get_n_leaves()

    print('test score: ', round(test_score,2))
    print('train score: ', round(train_score,2))
    print('tree depth: ', depth)
    print('terminal nodes (leaves): ', num_leaves)
    # print(tree.decision_path(X_test.iloc[0].values.reshape(1,16)))

def part_e(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=800)

    tree = DecisionTreeClassifier()

    tree.fit(X_train, y_train)
    preds = tree.predict(X_test)
    # cm = confusion_matrix(y_test, preds)
    #
    # sns.heatmap(cm)
    # plt.show()

    print(classification_report(y_test, preds))

def part_g(X, y):
    ## Instantiate the decision tree
    tree = DecisionTreeClassifier()
    best_score=0

    ## Create set of test alpha values and empty cv scores for training and testing
    max_size_value = np.arange(1,100)
    cv_score_depth_test = []
    cv_score_depth_train = []

    ## iteracte of alpha and score the training and test results
    for l in max_size_value:
        print(l)
        tree.set_params(max_depth = l)
        scores = cross_validate(tree, X, y, return_train_score=True, return_estimator=True)
        cv_score_depth_test.append(scores['test_score'].mean())
        cv_score_depth_train.append(scores['train_score'].mean())


    ## Plot the training and test scores
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(max_size_value, cv_score_depth_test, label = 'test score')
    ax.plot(max_size_value, cv_score_depth_train, label = 'training score')

    ax.set_xlabel(r'Max depth', fontsize = 'large')
    ax.set_ylabel('Score (R -squared)', fontsize = 'large')
    ax.set_title('Model performance as max length is increased', fontsize = 'xx-large')
    ax.legend(fontsize = 'large')

    plt.savefig(os.path.join(IMAGE_DIR, 'q9_tree_depth_plot.png'))
    plt.show()
    plt.close()


def main():
    ## Load dataframe
    oj_df = pd.read_csv(os.path.join(DATA_DIR, "OJ.csv"))
    print(oj_df.columns)


    # oj_df = pd.get_dummies(oj_df, 'Store7', drop_first=True)
    # print(oj_df.columns)


    y = oj_df['Purchase']
    X = oj_df.drop(['Purchase', 'Store7'], axis=1)

    ##Preprocessing

    # part_b(X,y)
    # part_e(X,y)
    part_g(X, y)

if __name__ == "__main__":
    main()

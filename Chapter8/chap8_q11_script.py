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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter8')

def part_b(X, y):
    X_cols = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, random_state=101)

    boost = xgb.XGBClassifier(n_estimators=1000, reg_alpha=0.01, random_state=101)
    boost.fit(X_train,y_train)
    print('test set score: ', boost.score(X_test, y_test))

    feat_import = list(zip(X_cols, boost.feature_importances_))
    feat_import.sort(key=lambda tup: tup[1], reverse=True)
    for feat in feat_import[0:5]:
        print("{0:<20}:{1:<20}".format(feat[0], feat[1]))

def part_c(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, random_state=101)

    boost = xgb.XGBClassifier(n_estimators=1000, reg_alpha=0.01, random_state=101)
    boost.fit(X_train,y_train)
    boost_prob_preds = boost.predict_proba(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_prob_preds = knn.predict_proba(X_test)

    logr = LogisticRegression()
    logr.fit(X_train, y_train)
    logr_prob_preds = logr.predict_proba(X_test)

    threshold = 0.2
    # define vectorized sigmoid
    threshold_v = np.vectorize(lambda x: 'No' if x < threshold else 'Yes')
    boost_prob_preds = threshold_v(boost_prob_preds[:,1])
    knn_prob_preds = threshold_v(knn_prob_preds[:,1])
    logr_prob_preds = threshold_v(logr_prob_preds[:,1])

    # print('XGBoost confusion matrix')
    # print(confusion_matrix(y_test.values, boost_prob_preds))
    # print('KNN confusion matrix')
    # print(confusion_matrix(y_test.values, knn_prob_preds))
    # print('Logistic Regression confusion matrix')
    # print(confusion_matrix(y_test.values, logr_prob_preds))

    boost_cm = confusion_matrix(y_test.values, boost_prob_preds)
    knn_cm = confusion_matrix(y_test.values, knn_prob_preds)
    logr_cm = confusion_matrix(y_test.values, logr_prob_preds)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    sns.heatmap(boost_cm, annot=True, fmt="d", ax = ax1) #annot=True to annotate cells
    sns.heatmap(knn_cm, annot=True, fmt="d", ax = ax2)
    sns.heatmap(logr_cm, annot=True, fmt="d", ax = ax3)

    # labels, title and ticks
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title('XGBoost Confusion Matrix')
    ax1.xaxis.set_ticklabels(['No', 'Yes'])
    ax1.yaxis.set_ticklabels(['No', 'Yes'])

    ax2.set_xlabel('Predicted labels')
    ax2.set_ylabel('True labels')
    ax2.set_title('KNN Confusion Matrix (N=5)')
    ax2.xaxis.set_ticklabels(['No', 'Yes'])
    ax2.yaxis.set_ticklabels(['No', 'Yes'])

    ax3.set_xlabel('Predicted labels')
    ax3.set_ylabel('True labels')
    ax3.set_title('Logistic Regression Confusion Matrix')
    ax3.xaxis.set_ticklabels(['No', 'Yes'])
    ax3.yaxis.set_ticklabels(['No', 'Yes'])

    fig.suptitle('Classifier Confusion Matricies for 20% threshold')

    plt.savefig('q11_confusion_matrices.png')
    plt.show()

def main():
    ## Load dataframe
    caravan_df = pd.read_csv(os.path.join(DATA_DIR, "caravan.csv"))
    # print(caravan_df.info())

    y = caravan_df['Purchase']
    X = caravan_df.drop('Purchase', axis=1)
    X_cols = X.columns

    part_b(X, y)
    part_c(X, y)



if __name__ == "__main__":
    main()

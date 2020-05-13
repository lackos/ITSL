import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter9')

np.random.seed(101)

def main():
    oj_df = pd.read_csv(os.path.join(DATA_DIR, 'OJ.csv'))

    y = oj_df['Purchase']
    X = oj_df.drop(['Purchase', 'Store7'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800)

    linear_svm = SVC(kernel='linear', C=10)
    linear_svm.fit(X_train, y_train)

    print(linear_svm.score(X_test, y_test))
    cv_score_C_test = []
    cv_score_C_train = []
    C_values = np.arange(10, 100, 0.1)
    # C_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    for c in C_values:
        print(c)
        linear_svm.set_params(C=c)
        scores = cross_validate(linear_svm, X, y, return_train_score=True)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_C_test.append(scores['test_score'].mean())
        cv_score_C_train.append(scores['train_score'].mean())

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(C_values, cv_score_C_test, label='test score')
    ax1.plot(C_values, cv_score_C_train, label='train_score')

    ax1.set_xlabel('C')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for C')
    ax1.legend(fontsize = 'large')
    # ax1.set_xscale('log')
    plt.show()



if __name__ == "__main__":
    main()

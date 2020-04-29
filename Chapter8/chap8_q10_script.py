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

import xgboost as xgb

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter8')

def part_c(X,y):
    alpha_values = np.arange(0, 10, 0.001)
    lambda_values = np.arange(0, 10, 0.001)
    # alpha_values = [np.power(10.0,x) for x in np.arange(-10,1,1)]
    # lambda_values = [np.power(10.0,x) for x in np.arange(-10,1,1)]
    cv_score_alpha_test = []
    cv_score_alpha_train = []
    cv_score_lambda_test = []
    cv_score_lambda_train = []

    boost = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')

    # boost.fit(X,y)
    # preds = boost.predict(X)
    # print(preds)

    boost.set_params(reg_lambda=0)

    for alpha in alpha_values:
        print(alpha)
        boost.set_params(reg_alpha=alpha)
        scores = cross_validate(boost, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_alpha_test.append(scores['test_score'].mean())
        cv_score_alpha_train.append(scores['train_score'].mean())

    boost.set_params(reg_alpha=0)

    for lam in lambda_values:
        print(lam)
        boost.set_params(reg_lambda=lam)
        scores = cross_validate(boost, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_lambda_test.append(scores['test_score'].mean())
        cv_score_lambda_train.append(scores['train_score'].mean())

    ## Plot the results of each cv loop. Both training and testing.
    fig, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols=1, figsize=(12,12))
    ax1.plot(alpha_values, cv_score_alpha_test, label='test score')
    ax1.plot(alpha_values, cv_score_alpha_train, label='train_score')

    ax2.plot(lambda_values, cv_score_lambda_test, label='test score')
    ax2.plot(lambda_values, cv_score_lambda_train, label='train_score')

    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for alpha (l1-regularization)')
    ax1.legend(fontsize = 'large')

    ax2.set_xlabel('lambda')
    ax2.set_ylabel('Score')
    ax2.set_title('Model performance for lambda (l2-regularization)')
    ax2.legend(fontsize = 'large')

    plt.savefig(os.path.join(IMAGE_DIR, 'q10_regularization_performance.png'))
    plt.show()

def main():
    ## Load dataframe
    hitters_df = pd.read_csv(os.path.join(DATA_DIR, "Hitters.csv"))
    print('columns: ', list(hitters_df.columns))
    # print(hitters_df.info())

    ## Drop the rows where salary is NaN
    hitters_df.dropna(subset=['Salary'], inplace=True)
    print(hitters_df.isna().sum())

    ## Log transform Salary
    hitters_df['log_Salary'] = hitters_df['Salary'].apply(lambda x: np.log(x))
    print('columns: ', list(hitters_df.columns))

    ## One-hot encode the object columns
    hitters_df = pd.get_dummies(hitters_df, ['League', 'Division', 'NewLeague'], drop_first=True)

    y = hitters_df['log_Salary']
    X = hitters_df.drop(['Salary', 'log_Salary'], axis=1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=200)

    part_c(X,y)

if __name__ == "__main__":
    main()

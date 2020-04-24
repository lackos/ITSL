import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

import itertools

from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter6')

def exploration(df):
    for col in df.columns:
        print(col)
    ## Number of na values
    print(df.isna().sum())
    ## Column datatypes
    print(df.dtypes)

def part_b(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        ## Ordinary Least Squares model
        X_train = X_train.values
        y_train = y_train.values

        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        ols = sm.OLS(y_train, X_train).fit()
        # print(ols.summary())
        predictions = ols.predict(X_test)
        error = mean_squared_error(y_test, predictions, squared=False)
        print("least squares test error (RMSE): ", round(error,2))

def part_c(X, y):
    ## Ridge Regression model

    ## Set baseline RMSE score
    best_score = 10000

    ## Set empty cv_scores for plot
    cv_scores_ridge = []

    ## Instantiate the ridge regressor
    ridge = Ridge(tol=0.1, normalize=True)

    ## Loop over alpha values to find optimal cross-validated test error
    for lamb in np.arange(1,4000,1):
        ridge.set_params(alpha=lamb)
        scores = cross_validate(ridge, X, y, scoring='neg_root_mean_squared_error')
        cv_scores_ridge.append(scores['test_score'].mean())

        ## Test if cv score beat previous best
        if np.abs(scores['test_score'].mean()) < best_score:
            ridge.fit(X,y)
            best_score = scores['test_score'].mean()
            best_lamb = lamb
            best_model_coef = ridge.coef_
    print("Best Ridge Test score: ", round(np.abs(best_score),2))
    print("Ridge coefficients", best_model_coef)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax.plot(np.abs(cv_scores_ridge), lw=2)

    ax.set_xlabel('lambda', fontsize='medium')
    ax.set_ylabel('test RMSE', fontsize='medium')

    ax.set_title("R_squared vs lambda scale in Ridge", fontsize='large')
    plt.savefig(os.path.join(IMAGE_DIR, "q9_ridge_lambda_plot.png"))
    plt.show()

def part_d(X, y):
    ## Lasso Regression model

    ## Set baseline RMSE score
    best_score = 10000

    ## Set empty cv_scores for plot
    cv_scores_lasso = []

    ## Instantiate the ridge regressor
    lasso = Lasso(tol=0.1, normalize=True)

    ## Loop over alpha values to find optimal cross-validated test error
    for lamb in np.arange(10,4000,1):
        lasso.set_params(alpha=lamb)
        scores = cross_validate(lasso, X, y, scoring='neg_root_mean_squared_error')
        cv_scores_lasso.append(scores['test_score'].mean())

        ## Test if cv score beat previous best
        if np.abs(scores['test_score'].mean()) < best_score:
            lasso.fit(X,y)
            best_score = scores['test_score'].mean()
            best_lamb = lamb
            best_model_coef = lasso.coef_
    unique, counts = np.unique(best_model_coef, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print("Best Lasso Test score: ", round(np.abs(best_score),2))
    print(count_dict)
    # print("Lasso coefficients", best_model_coef)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax.plot(np.abs(cv_scores_lasso), lw=2)

    ax.set_xlabel('lambda', fontsize='medium')
    ax.set_ylabel('test RMSE', fontsize='medium')

    ax.set_title("test RMSE vs lambda scale in Lasso", fontsize='large')
    plt.savefig(os.path.join(IMAGE_DIR, "q9_lasso_lambda_plot_wfe.png"))
    plt.show()

def part_e(X, y):
    ## Principal Component Regression Model

    ## Set baseline RMSE score
    best_score = 10000

    ## Scale predictors and break down data set into principal components
    scaler = StandardScaler()
    pca = PCA()

    X_scaled = scaler.fit_transform(X)
    X_PCA = pca.fit_transform(X_scaled)

    ## Set empty score list
    cv_scores_PCR = []

    ## Instantiate the ordinary linear regressor
    ols = LinearRegression()

    ## Iterate over number of principal components used
    for i in range(1,X_PCA.shape[1]):
        ## Select the first i principal components to train
        X_train = X_PCA[:,0:i]
        scores = cross_validate(ols, X_train, y, scoring='neg_root_mean_squared_error')
        cv_scores_PCR.append(scores['test_score'].mean())
        ## Test if cv score beat previous best
        if np.abs(scores['test_score'].mean()) < best_score:
            best_score = np.abs(scores['test_score'].mean())
            best_num_PC = i
    print("Best PCR Test score: ", round(np.abs(best_score),2))
    print("Number of Principal Components", best_num_PC)
    ## Note fix train-test contamination

def part_f(X, y):
    ## Partial Least Squares Approach

    ## Set baseline RMSE score
    best_score = 10000

    ## Instantiate preprocessing
    pls = PLSRegression()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ## Set empty score list
    cv_scores_PLS = []
    for i in range(2,64):
        pls.set_params(n_components=i)
        scores = cross_validate(pls, X_scaled, y, scoring='neg_root_mean_squared_error')
        if np.abs(scores['test_score'].mean()) < best_score:
            best_score = np.abs(scores['test_score'].mean())
            best_num_PC = i
    print("Best PLS Test score: ", round(np.abs(best_score),2))
    print("Number of Latent variables", best_num_PC)


def main():
    ## Load dataframe
    college_df = pd.read_csv(os.path.join(DATA_DIR, "college.csv"))
    college_df = college_df.set_index('Unnamed: 0')
    # print(college_df)

    ## One hot encode the 'Private' column
    college_df = pd.get_dummies(college_df, drop_first=True)

    ## For each original numerical predictor X, create cols X^2, X^3 and X^4
    # for col in college_df.columns:
    #     if col == 'Private_Yes' or col == 'Apps':
    #         pass
    #     else:
    #         for i in [2,3,4]:
    #             new_col = col + '_' + str(i)
    #             college_df[new_col] = college_df[col].apply(lambda x: np.power(x,i))

    # for col in college_df.columns:
    #     print(col)

    y = college_df['Apps']
    X = college_df.drop('Apps', axis=1)

    # print(y)
    # print(X)

    # part_b(X, y)
    part_c(X, y)
    # part_d(X, y)
    # part_e(X, y)
    # part_f(X, y)


if __name__ == "__main__":
    main()

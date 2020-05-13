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
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter7')

def gen_predictor_array(X, order_list):
    X_array = np.zeros((X.shape[0],1))
    for integer in order_list:
        P = np.power(X,integer)
        P = P.reshape(P.shape[0], 1)
        X_array = np.append(X_array, P, axis=1)
    X_array = X_array[:,1:]
    return X_array

def part_a(X, y):
    ## Instantiate the linear regressor estimator
    ols = LinearRegression()

    ## Set baseline RMSE score
    best_score = 10000

    ## Create array to add predictors to in loop
    X_array = np.zeros((X.shape[0],1))

    ## Create score lists for plotting
    cv_score_order_test = []
    cv_score_order_train = []

    ## List of polynomial order to try and regress to
    order_list = [1,2,3,4,5,6,7,8,9,10]

    ## For loop to go over the polynomial
    for i in order_list:
        # print(i)
        P = np.power(X.values,i)
        P = P.reshape(P.shape[0], 1)
        X_array = np.append(X_array, P, axis=1)
        X_train = X_array[:,1:]
        scores = cross_validate(ols, X_train, y, scoring='neg_root_mean_squared_error', return_train_score=True)
        cv_score_order_test.append(scores['test_score'].mean())
        cv_score_order_train.append(scores['train_score'].mean())
        # print(np.abs(scores['test_score'].mean()))
        ## Test if cv score beat previous best
        if np.abs(scores['test_score'].mean()) < best_score:
            ols.fit(X_train,y)
            best_score = np.abs(scores['test_score'].mean())
            best_model_coef = ols.coef_
            best_model = ols
            best_order = i

    print("Best order is: ", best_order)
    print("Best score is: ", best_score)

    poly_fit_plot(X,y,best_model,best_order)

    fig, ax = plt.subplots(figsize = (12,6))
    ax.plot(order_list, np.abs(cv_score_order_test), lw='3', marker='.', ms='15', label='Mean Testing score')
    ax.plot(order_list, np.abs(cv_score_order_train), lw='3', marker='.', ms='15', label='Mean Training score')

    ax.set_xlabel('Polynomial Order', fontsize = 'large')
    ax.set_ylabel('RMSE (1000$)', fontsize = 'large')
    ax.set_title('Wage vs Age: polynomial fit test errors', fontsize='xx-large')
    ax.legend(fontsize = 'large')
    # plt.show()
    # plt.savefig(os.path.join(IMAGE_DIR,'q1_poly_reg_error_plot.png'))
    # print(X_train)
    plt.close()

def poly_fit_plot(X,y,model,order):

    x_range = np.arange(15,90,1)
    x_array = x_range.reshape(x_range.shape[0],1)
    # print(x_range)
    for i in np.arange(1,order + 1,1):
        # print(i)
        P = np.power(x_range,i)
        P = P.reshape(P.shape[0], 1)
        x_array = np.append(x_array, P, axis=1)
    X_theo = x_array[:,1:]
    y_theo = model.predict(X_theo)
    # print(X_theo)
    # print(y_theo)
    fig, ax = plt.subplots(figsize = (12,6))
    ax.scatter(X, y, label='data points')
    ax.plot(x_range, y_theo, lw='3', color='r', label='polynomial fit')

    ax.set_xlabel('Age', fontsize = 'large')
    ax.set_ylabel('Wage (1000$)', fontsize = 'large')
    ax.set_title('Wage vs Age (Polynomial Fit)', fontsize='xx-large')
    ax.legend(fontsize = 'large')
    plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, 'q1,poly_reg_fit.png'))
    plt.close()



def main():
    ## Load dataframe
    wage_df = pd.read_csv(os.path.join(DATA_DIR, "wage.csv"))
    X = wage_df['age']
    y = wage_df['wage']

    # part_a(X,y)
    # poly_fit_plot(X,y,X,5)
    # print(wage_df)

if __name__ == "__main__":
    main()

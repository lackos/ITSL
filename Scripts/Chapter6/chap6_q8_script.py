import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

import itertools

from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.utils import shuffle

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter6')

np.random.seed(100)

def bss(pred_dict, Y, measure='adj_Rsquared'):
    """
    Function to perform best subset selection for a given response and dictionary of predictors. Returns the model with the best subset of parameters fitted

    Keyword Arguments-
    pred_dict: dictionary of predictors to be tested.
    Y: Response variable.
    measure: Which metric to use to determine best subset. Can be
        'adj_Rsquared'
        'BIC'
        'AIC'
    """
    ## Set initial metrics to improve
    adjusted_R2 = 0
    BIC = 1000000
    AIC = 1000000

    for k in range(0, len(pred_dict)):
        ## For each k in p predictors find pCk and set them as a list
        pck = list(list(itertools.combinations(pred_dict.keys(), k)))
        for i in pck:
            ## For each of these permutations
            if i == ():
                ## Ignore permutation if subset is empty
                pass
            else:
                ## Set empty numpy array for training
                X_train = np.zeros((pred_dict[i[0]].shape[0],1))
                for j in range(0,len(i)):
                    ## Add each predictor in permutation to the training array
                    pred_dict[i[j]] = pred_dict[i[j]].reshape(pred_dict[i[j]].shape[0], 1)
                    X_train = np.append(X_train, pred_dict[i[j]], axis=1)
                ## Drop the zero column and add_constant to OLS
                X_train = X_train[:,1:]
                X_train = sm.add_constant(X_train)
                # print(X_train)
                # X_train = pred_dict{}

                ## Fit the model and train with the training array
                model = sm.OLS(Y, X_train).fit()
                # print(model.summary())
                # print(model.rsquared)

                ## Given the measure update the best subset and save the model
                if model.rsquared_adj > adjusted_R2 and measure == 'adj_Rsquared':
                    adjusted_R2=model.rsquared_adj
                    best_selection = i
                    best_model = model
                    best_X = X_train
                elif model.bic < BIC and measure == 'BIC':
                    BIC=model.bic
                    best_selection = i
                    best_model = model
                    best_X = X_train
                elif model.aic < AIC and measure == 'AIC':
                    AIC=model.aic
                    best_selection = i
                    best_model = model
                    best_X = X_train
    print("Best subset of predictors", best_selection)
    return best_model

def gen_predictor_array(X, order_list):
    X_array = np.zeros((X.shape[0],1))
    for integer in order_list:
        P = np.power(X,integer)
        P = P.reshape(P.shape[0], 1)
        X_array = np.append(X_array, P, axis=1)
    X_array = X_array[:,1:]
    return X_array

def part_ab():
    ## Generate the simulated data
    X = np.random.randn(100)
    err = np.random.randn(100)

    ## Set the estimators
    beta_0 = 10
    beta_1 = 20
    beta_2 = 20
    beta_3 = 3

    ## Generate the response
    Y = beta_0 + beta_1*X + beta_2*X**2 + beta_3*X**3 + err
    return X, Y

def part_c(X, Y):
    ## Dictionary of predictors
    predictor_dict = {'X_1':X,
                      'X_2':X**2,
                      'X_3':X**4,
                      'X_4':X**4,
                      'X_5':X**5,
                      'X_6':X**6,
                      'X_7':X**7,
                      'X_8':X**8,
                      'X_9':X**9,
                      'X_10':X**10}
    ars_model = bss(predictor_dict, Y, measure='adj_Rsquared')
    bic_model = bss(predictor_dict, Y, measure='BIC')
    aic_model = bss(predictor_dict, Y, measure='AIC')

    X_cont = np.arange(0,10,0.005)
    X_array = gen_predictor_array(X_cont, [1,2,5,7,9])
    X_array = sm.add_constant(X_array)

    Y_ars = ars_model.predict(X_array)
    Y_BIC = bic_model.predict(X_array)
    Y_AIC = aic_model.predict(X_array)

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1, 0.8,0.8])
    sns.scatterplot(X, Y, ax=ax)
    sns.scatterplot(X_cont, Y_ars)
    sns.scatterplot(X_cont, Y_AIC)
    sns.scatterplot(X_cont, Y_BIC)
    plt.show()

def part_e(X, Y):
    X_train = gen_predictor_array(X, range(1,11))
    lin_reg = LinearRegression()
    # lin_reg.fit(X_train, Y)
    # print(lin_reg.score(X_train, Y))
    # print(cross_val_score(lin_reg, X_train, Y).mean())

    lasso = Lasso()
    # lasso.fit(X_train, Y)
    # print(cross_val_score(lasso, X_train, Y).mean())

    cv_scores_lasso = []
    for lamb in np.arange(0,2000,1):
        lasso = Lasso(alpha=lamb)
        # lasso.fit(X_train, Y)
        cv_scores_lasso.append(cross_val_score(lasso, X_train, Y).mean())
    print(cv_scores_lasso)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax.plot(cv_scores_lasso, lw=2)

    ax.set_xlabel('lambda', fontsize='medium')
    ax.set_ylabel('R_squared', fontsize='medium')

    ax.set_title("R_squared vs lambda scale in Lasso", fontsize='large')
    plt.savefig(os.path.join(IMAGE_DIR, "lasso_lambda_plot.png"))
    plt.show()

def part_f():
    ## Lasso model
    X = np.random.randn(100)
    beta_0 = 10
    beta_7 = 30
    Y = beta_0 + beta_7*X**7  + np.random.randn(100)
    X_train = gen_predictor_array(X, range(1,11))
    # X_train = sm.add_constant(X_train)


def main():
    X, Y = part_ab()
    predictor_dict = {'X_1':X,
                      'X_2':X**2,
                      'X_3':X**4,
                      'X_4':X**4,
                      'X_5':X**5,
                      'X_6':X**6,
                      'X_7':X**7,
                      'X_8':X**8,
                      'X_9':X**9,
                      'X_10':X**10}
    fss(predictor_dict)
    # model = part_c(X,Y)
    # bss({'X_1':X, 'X_2':X**2 , 'X_3':X**4, 'X_4':X**4, 'X_5':X**5}, Y)
    # part_e(X, Y)


if __name__ == "__main__":
    main()

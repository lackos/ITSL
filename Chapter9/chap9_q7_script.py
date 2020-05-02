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

def plot_score_vs_param(model,X, y, para, param_values):
    ## Set baseline score
    best_score = 0

    ## Set empty list for storing training and test scores
    cv_score_test = []
    cv_score_train = []
    values = param_values
    # C_values = [np.power(10.0,x) for x in np.arange(0,7,1)]
    for val in values:
        print(para + '_value: ' + str(val))
        if para == 'C':
            model.set_params(C=val)
        if para == 'gamma':
            model.set_params(gamma=val)
        if para == 'degree':
            model.set_params(degree=val)
        scores = cross_validate(model, X, y, return_train_score=True, n_jobs=-1)
        print('test score: ', scores['test_score'].mean())
        print('train score: ', scores['train_score'].mean())
        cv_score_test.append(scores['test_score'].mean())
        cv_score_train.append(scores['train_score'].mean())

        ## If the iteraction is the current best perfomer, save the parameters and model
        if scores['test_score'].mean() > best_score:
            best_score = scores['test_score'].mean()
            model.fit(X,y)
            best_model = model
            best_value = val

    print('Best value of cost (C): ', best_value)
    print('Best test set score: ', round(best_score, 3))

    results = {}
    results['best_test_score'] = best_score
    results['best_model'] = best_model
    results['best_value'] = best_value
    results['train_scores'] = cv_score_train
    results['test_scores'] = cv_score_test

    return results

def part_b(X, y):
    ## Instantiate the linear SVC
    linear_svm = SVC(kernel='linear')
    C_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]

    C_results = plot_score_vs_param(linear_svm , X, y, para='C', param_values=C_values)

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(C_values, C_results['test_scores'], label='test score')
    ax1.plot(C_values, C_results['train_scores'], label='train_score')

    ax1.set_xlabel('C')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for C')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q7_linear_C_plot.png'))
    plt.show()
    plt.close()

def part_c_poly(X, y):
    poly_svm = SVC(kernel='poly', max_iter=100000000)

    ## Set the values to iterate over
    ### Polynomial degrees
    degree_values = [1,2,3,4,5,6]
    ### Gamma Values
    # gamma_values = np.arange(0.001, 1, 0.01)
    gamma_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    ### Cost Values
    # C_values = np.arange(0.001, 5, 0.01)
    C_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]

    ## Set up figure to plot train, test scores
    fig, axs = plt.subplots(nrows=len(degree_values), ncols=2, figsize=(12,32))
    ## For each degree produce a new row in the plot
    for idx, deg in enumerate(degree_values):
        print("Degree: ", deg)
        ## Update model with new polynomial degree
        poly_svm.set_params(degree=deg)
        ## Perform CV with gamma values
        gamma_results = plot_score_vs_param(poly_svm, X, y, para='gamma', param_values=gamma_values)
        ## Update model with best gamma value
        poly_svm.set_params(gamma=gamma_results['best_value'])
        ## Perform CV with C values
        C_results = plot_score_vs_param(poly_svm, X, y, para='C', param_values=C_values)

        ## Update axes with new plots
        axs[idx,0].plot(gamma_values, gamma_results['test_scores'], label='test score')
        axs[idx,0].plot(gamma_values, gamma_results['train_scores'], label='train_score')
        axs[idx,0].set_xlabel('gamma')
        axs[idx,0].set_ylabel('Score')
        axs[idx,0].set_title('Model performance for gamma (Degree {})'.format(deg))
        axs[idx,0].legend(fontsize = 'large')
        axs[idx,0].set_xscale('log')

        axs[idx,1].plot(C_values, C_results['test_scores'], label='test score')
        axs[idx,1].plot(C_values, C_results['train_scores'], label='train_score')
        axs[idx,1].set_xlabel('C')
        axs[idx,1].set_ylabel('Score')
        axs[idx,1].set_title('Model performance for C (Degree: {0}, gamma: {1:.2})'.format(deg,gamma_results['best_value']))
        axs[idx,1].legend(fontsize = 'large')
        axs[idx,1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'q7_poly_cv_plots.png'), dpi = 300)
    plt.show()
    plt.close()

def part_c_rbf(X, y):
    rbf_svm = SVC(kernel='rbf', max_iter=100000000)

    ## Set the values to iterate over
    ### Gamma Values
    # gamma_values = np.arange(0.001, 1, 0.01)
    gamma_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    ### Cost Values
    # C_values = np.arange(0.001, 5, 0.01)
    C_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    ## Perform CV with gamma values
    gamma_results = plot_score_vs_param(rbf_svm, X, y, para='gamma', param_values=gamma_values)
    ## Update model with best gamma value
    rbf_svm.set_params(gamma=gamma_results['best_value'])
    ## Perform CV with C values
    C_results = plot_score_vs_param(rbf_svm, X, y, para='C', param_values=C_values)

    ## Update axes with new plots
    ax1.plot(gamma_values, gamma_results['test_scores'], label='test score')
    ax1.plot(gamma_values, gamma_results['train_scores'], label='train_score')
    ax1.set_xlabel('gamma')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for gamma')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    ax2.plot(C_values, C_results['test_scores'], label='test score')
    ax2.plot(C_values, C_results['train_scores'], label='train_score')
    ax2.set_xlabel('C')
    ax2.set_ylabel('Score')
    ax2.set_title('Model performance for C')
    ax2.legend(fontsize = 'large')
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'q7_rbf_cv_plots.png'))
    plt.show()
    plt.close()

def main():
    ## Load the auto dataset
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    ## Create the classification column based on the median mpg
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)
    ## Clean up remaining data
    auto_df = auto_df[auto_df['horsepower'] != '?']
    auto_df['horsepower'] = auto_df['horsepower'].apply(lambda x: float(x))

    print(auto_df.info())
    print(auto_df.isna().sum())

    y = auto_df['mpg01']
    X = auto_df.drop(['mpg01', 'name'], axis=1)

    # part_b(X, y)
    # part_c_poly(X, y)
    part_c_rbf(X, y)


if __name__ == "__main__":
    main()

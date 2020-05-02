import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVC

import xgboost as xgb

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter9')

np.random.seed(102)

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

def data_plot(x_1, x_2, y):
    ## Plot the data with labels
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    sns.scatterplot(x=x_1, y=x_2, hue=y, ax=ax)

    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_title('Data set with quasi-linear boundary with f')

    # plt.savefig(os.path.join(IMAGE_DIR,'q1_class_plot.png'))
    plt.show()
    plt.close()

def linear_svm(X, y):
    ## Instantiate the linear SVC
    lin_svm = SVC(kernel='linear')
    C_values = np.arange(0.001, 5, 0.001)

    ## Set baseline score
    best_score = 0

    ## Set empty list for storing training and test scores
    cv_score_C_test = []
    cv_score_C_train = []
    C_values = np.arange(0.001, 5, 0.001)
    # C_values = [np.power(10.0,x) for x in np.arange(0,7,1)]
    for c in C_values:
        print(c)
        lin_svm.set_params(C=c)
        scores = cross_validate(lin_svm, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_C_test.append(scores['test_score'].mean())
        cv_score_C_train.append(scores['train_score'].mean())

        ## If the iteraction is the current best perfomer, save the parameters and model
        if scores['test_score'].mean() > best_score:
            best_score = scores['test_score'].mean()
            lin_svm.fit(X,y)
            best_model = lin_svm
            best_C_value = c

    print('Best value of cost (C): ', best_C_value)
    print('Best test set score: ', round(best_score, 3))

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(C_values, cv_score_C_test, label='test score')
    ax1.plot(C_values, cv_score_C_train, label='train score')

    ax1.set_xlabel('C')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for gamma')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    # plt.savefig(os.path.join(IMAGE_DIR, 'q4_linear_C.png'))
    plt.show()
    plt.close()

def poly_svm(X, y):
    ## Instantiate the SVC with polynomial kernel
    ## Maximum iteractions set as finding optimal solution will be difficult for wrong fits.
    poly_svm = SVC(kernel='poly', max_iter=10000000)

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
    plt.savefig(os.path.join(IMAGE_DIR, 'q4_poly_cv_plots.png'), dpi = 300)
    plt.show()
    plt.close()
    # results = plot_score_vs_param(poly_svm, X, y, para='C', param_values=C_values)
    #
    #
    #
    #
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    # ax1.plot(C_values, results['test_scores'], label='test score')
    # ax1.plot(C_values, results['train_scores'], label='train_score')
    #
    # ax1.set_xlabel('C')
    # ax1.set_ylabel('Score')
    # ax1.set_title('Model performance for gamma')
    # ax1.legend(fontsize = 'large')
    # ax1.set_xscale('log')
    #
    # # plt.savefig(os.path.join(IMAGE_DIR, 'q4_poly_C.png'))
    # plt.show()
    # plt.close()
    #
    # best_score = 0
    #
    # # Set empty list for storing training and test scores
    # cv_score_gamma_test = []
    # cv_score_gamma_train = []
    # gamma_values = np.arange(0.001, 1, 0.001)
    #
    # # gamma_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    # for g in gamma_values:
    #     print(g)
    #     poly_svm.set_params(gamma=g)
    #     scores = cross_validate(poly_svm, X, y, return_train_score=True, n_jobs=-1)
    #     print(scores['test_score'].mean())
    #     print(scores['train_score'].mean())
    #     cv_score_gamma_test.append(scores['test_score'].mean())
    #     cv_score_gamma_train.append(scores['train_score'].mean())
    #     ## If the iteraction is the current best perfomer, save the parameters and model
    #     if scores['test_score'].mean() > best_score:
    #         best_score = scores['test_score'].mean()
    #         poly_svm.fit(X,y)
    #         best_model = poly_svm
    #         best_gamma_value = g
    #
    # print('Best value of gamma: ', best_gamma_value)
    # print('Best test set score: ', round(best_score, 3))
    #
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    # ax1.plot(gamma_values, cv_score_gamma_test, label='test score')
    # ax1.plot(gamma_values, cv_score_gamma_train, label='train_score')
    #
    # ax1.set_xlabel('gamma')
    # ax1.set_ylabel('Score')
    # ax1.set_title('Model performance for gamma')
    # ax1.legend(fontsize = 'large')
    # ax1.set_xscale('log')
    #
    # plt.savefig(os.path.join(IMAGE_DIR, 'q4_poly_gamma.png'))
    # plt.show()
    # plt.close()
    #
    # cv_score_C_test = []
    # cv_score_C_train = []
    # C_values = np.arange(0.001, 5, 0.001)
    # # C_values = [np.power(10.0,x) for x in np.arange(0,7,1)]
    # poly_svm.set_params(gamma=0.01)
    # for c in C_values:
    #     print(c)
    #     poly_svm.set_params(C=c)
    #     scores = cross_validate(poly_svm, X, y, return_train_score=True, n_jobs=-1)
    #     print(scores['test_score'].mean())
    #     print(scores['train_score'].mean())
    #     cv_score_C_test.append(scores['test_score'].mean())
    #     cv_score_C_train.append(scores['train_score'].mean())
    #     if scores['test_score'].mean() > best_score:
    #         best_score = scores['test_score'].mean()
    #         poly_svm.fit(X,y)
    #         best_model = poly_svm
    #         best_C_value = c
    #
    #
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    # ax1.plot(C_values, cv_score_C_test, label='test score')
    # ax1.plot(C_values, cv_score_C_train, label='train_score')
    #
    # ax1.set_xlabel('C')
    # ax1.set_ylabel('Score')
    # ax1.set_title('Model performance for gamma')
    # ax1.legend(fontsize = 'large')
    # ax1.set_xscale('log')
    #
    # # plt.savefig(os.path.join(IMAGE_DIR, 'q4_poly_C.png'))
    # plt.show()
    # plt.close()

def rbf_svm(X, y):
    rbf_svm = SVC(kernel='rbf')

    ## Set the values to iterate over
    ### Gamma Values
    # gamma_values = np.arange(0.001, 1, 0.01)
    gamma_values = [np.power(10.0,x) for x in np.arange(-7,2,1)]
    ### Cost Values
    # C_values = np.arange(0.001, 5, 0.01)
    C_values = [np.power(10.0,x) for x in np.arange(-7,3,1)]

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
    plt.savefig(os.path.join(IMAGE_DIR, 'q4_rbf_cv_plots.png'))
    plt.show()
    plt.close()

    # cv_score_gamma_test = []
    # cv_score_gamma_train = []
    # # gamma_values = np.arange(0.001, 1, 0.001)
    # gamma_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    # for gamma in gamma_values:
    #     print(gamma)
    #     rbf_svm.set_params(gamma=gamma)
    #     scores = cross_validate(rbf_svm, X, y, return_train_score=True, n_jobs=-1)
    #     print(scores['test_score'].mean())
    #     print(scores['train_score'].mean())
    #     cv_score_gamma_test.append(scores['test_score'].mean())
    #     cv_score_gamma_train.append(scores['train_score'].mean())
    #
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    # ax1.plot(gamma_values, cv_score_gamma_test, label='test score')
    # ax1.plot(gamma_values, cv_score_gamma_train, label='train_score')
    #
    # ax1.set_xlabel('gammaa')
    # ax1.set_ylabel('Score')
    # ax1.set_title('Model performance for gamma')
    # ax1.legend(fontsize = 'large')
    # ax1.set_xscale('log')
    #
    # plt.savefig(os.path.join(IMAGE_DIR, 'q4_rbf_gamma.png'))
    # plt.close()
    #
    # cv_score_C_test = []
    # cv_score_C_train = []
    # C_values = np.arange(0.001, 5, 0.001)
    # # C_values = [np.power(10.0,x) for x in np.arange(0,7,1)]
    # rbf_svm.set_params(gamma=0.01)
    # for c in C_values:
    #     print(c)
    #     rbf_svm.set_params(C=c)
    #     scores = cross_validate(rbf_svm, X, y, return_train_score=True, n_jobs=-1)
    #     print(scores['test_score'].mean())
    #     print(scores['train_score'].mean())
    #     cv_score_C_test.append(scores['test_score'].mean())
    #     cv_score_C_train.append(scores['train_score'].mean())
    #
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    # ax1.plot(C_values, cv_score_C_test, label='test score')
    # ax1.plot(C_values, cv_score_C_train, label='train_score')
    #
    # ax1.set_xlabel('C')
    # ax1.set_ylabel('Score')
    # ax1.set_title('Model performance for gamma')
    # ax1.legend(fontsize = 'large')
    # ax1.set_xscale('log')
    #
    # plt.savefig(os.path.join(IMAGE_DIR, 'q4_rbf_C.png'))
    # plt.close()

def main():
    ### Problem 4 of Introduction to Statistic Learning Chapter 9

    ## Define the two predictors and create a dataframe to pass to models
    x_1 = 10*np.random.uniform(size=50) - 5
    x_2 = 10*np.random.uniform(size=50) - 5
    X = np.column_stack((x_1, x_2))

    ## Create the labels
    y = []
    for i in range(len(x_2)):
        ## Non linear function
        arg = x_1[i]**2 - x_2[i]**2
        if arg > 0:
            y.append(1)
        else:
            y.append(-1)
    y = np.array(y)


    # data_plot(x_1, x_2, y)
    # linear_svm(X, y)
    # poly_svm(X, y)
    rbf_svm(X, y)
    # poly_svm(X, y)

    # class1_x = np.random.normal(loc=1, scale = 1, size=50)
    # class1_y = np.random.normal(loc=1, scale = 1, size=50)
    # class1_target = np.full((50,), 'red')
    #
    # # class1 = np.append(class1_x, class1_y, axis=1)
    # class1 = np.column_stack((class1_x, class1_y, class1_target))
    #
    # class2_x = np.random.normal(loc=-1, scale = 1.2, size=50)
    # class2_y = np.random.normal(loc=-1, scale = 1.2, size=50)
    # class2_target = np.full((50,), 'blue')
    # class2 = np.column_stack((class2_x, class2_y, class2_target))

    # data = np.append(class1, class2, axis=0)
    # np.random.shuffle(data)

    # fig, ax = plt.subplots(1,1)
    # ax.scatter(class1_x, class1_y, color='red')
    # ax.scatter(class2_x, class2_y**2, color='blue')
    # plt.show()
    # plt.close()



if __name__ == "__main__":
    main()

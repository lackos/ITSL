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

import xgboost as xgb

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter9')

def linear_svm(X, y):
    lin_svm = SVC(kernel='linear')

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

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(C_values, cv_score_C_test, label='test score')
    ax1.plot(C_values, cv_score_C_train, label='train_score')

    ax1.set_xlabel('C')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for gamma')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q4_linear_C.png'))
    plt.close()

def poly_svm(X, y):
    poly_svm = SVC(kernel='poly')
    cv_score_gamma_test = []
    cv_score_gamma_train = []
    # gamma_values = np.arange(0.001, 1, 0.001)
    gamma_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    for gamma in gamma_values:
        print(gamma)
        poly_svm.set_params(gamma=gamma)
        scores = cross_validate(poly_svm, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_gamma_test.append(scores['test_score'].mean())
        cv_score_gamma_train.append(scores['train_score'].mean())

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(gamma_values, cv_score_gamma_test, label='test score')
    ax1.plot(gamma_values, cv_score_gamma_train, label='train_score')

    ax1.set_xlabel('gammaa')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for gamma')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q4_poly_gamma.png'))
    plt.show()
    plt.close()

    cv_score_C_test = []
    cv_score_C_train = []
    C_values = np.arange(0.001, 5, 0.001)
    # C_values = [np.power(10.0,x) for x in np.arange(0,7,1)]
    poly_svm.set_params(gamma=0.01)
    for c in C_values:
        print(c)
        poly_svm.set_params(C=c)
        scores = cross_validate(poly_svm, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_C_test.append(scores['test_score'].mean())
        cv_score_C_train.append(scores['train_score'].mean())

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(C_values, cv_score_C_test, label='test score')
    ax1.plot(C_values, cv_score_C_train, label='train_score')

    ax1.set_xlabel('C')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for gamma')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q4_poly_C.png'))
    plt.show()
    plt.close()

def rbf_svm(X, y):
    rbf_svm = SVC(kernel='rbf')
    cv_score_gamma_test = []
    cv_score_gamma_train = []
    # gamma_values = np.arange(0.001, 1, 0.001)
    gamma_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    for gamma in gamma_values:
        print(gamma)
        rbf_svm.set_params(gamma=gamma)
        scores = cross_validate(rbf_svm, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_gamma_test.append(scores['test_score'].mean())
        cv_score_gamma_train.append(scores['train_score'].mean())

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(gamma_values, cv_score_gamma_test, label='test score')
    ax1.plot(gamma_values, cv_score_gamma_train, label='train_score')

    ax1.set_xlabel('gammaa')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for gamma')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q4_rbf_gamma.png'))
    plt.close()

    cv_score_C_test = []
    cv_score_C_train = []
    C_values = np.arange(0.001, 5, 0.001)
    # C_values = [np.power(10.0,x) for x in np.arange(0,7,1)]
    rbf_svm.set_params(gamma=0.01)
    for c in C_values:
        print(c)
        rbf_svm.set_params(C=c)
        scores = cross_validate(rbf_svm, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_C_test.append(scores['test_score'].mean())
        cv_score_C_train.append(scores['train_score'].mean())

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(C_values, cv_score_C_test, label='test score')
    ax1.plot(C_values, cv_score_C_train, label='train_score')

    ax1.set_xlabel('C')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for gamma')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q4_rbf_C.png'))
    plt.close()

def main():
    class1_x = np.random.normal(loc=1, scale = 1, size=50)
    class1_y = np.random.normal(loc=1, scale = 1, size=50)
    class1_target = np.full((50,), 'red')

    # class1 = np.append(class1_x, class1_y, axis=1)
    class1 = np.column_stack((class1_x, class1_y, class1_target))

    class2_x = np.random.normal(loc=-1, scale = 1.2, size=50)
    class2_y = np.random.normal(loc=-1, scale = 1.2, size=50)
    class2_target = np.full((50,), 'blue')
    class2 = np.column_stack((class2_x, class2_y, class2_target))

    data = np.append(class1, class2, axis=0)
    np.random.shuffle(data)

    X = data[:,0:2]
    y = data[:,-1]

    # rbf_svm(X, y)
    poly_svm(X, y)

    # X_train = X[:70,:]
    # y_train = y[:70]
    # X_test = X[70:,:]
    # y_test = y[70:]








    # fig, ax = plt.subplots(1,1)
    # ax.scatter(class1_x, class1_y, color='red')
    # ax.scatter(class2_x, class2_y**2, color='blue')
    # # plt.show()
    # plt.close()

if __name__ == "__main__":
    main()

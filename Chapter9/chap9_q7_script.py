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

def part_b(X, y):
    linear_svm = SVC(kernel='linear')
    cv_score_C_test = []
    cv_score_C_train = []
    # C_values = np.arange(0.01, 100, 0.01)
    C_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    for c in C_values:
        print(c)
        linear_svm.set_params(C=c)
        scores = cross_validate(linear_svm, X, y, return_train_score=True, n_jobs=-1)
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
    ax1.set_xscale('log')

    # plt.savefig(os.path.join(IMAGE_DIR, 'q4_poly_C.png'))
    plt.show()
    plt.close()

def part_c_Cvalue(X, y):
    poly_svm = SVC(kernel='poly')
    cv_score_C_test = []
    cv_score_C_train = []
    # C_values = np.arange(0.01, 100, 0.01)
    C_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
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
    ax1.set_title('Model performance for C')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q7_poly_C.png'))
    plt.show()
    plt.close()

def part_c_degree(X, y):
    poly_svm = SVC(kernel='poly', C=100)
    cv_score_deg_test = []
    cv_score_deg_train = []
    deg_values = np.arange(1, 10, 1)
    for d in deg_values:
        print(d)
        poly_svm.set_params(degree=d)
        scores = cross_validate(poly_svm, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_deg_test.append(scores['test_score'].mean())
        cv_score_deg_train.append(scores['train_score'].mean())

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(deg_values, cv_score_deg_test, label='test score')
    ax1.plot(deg_values, cv_score_deg_train, label='train_score')


    ax1.set_xlabel('C')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for different degrees')
    ax1.legend(fontsize = 'large')
    # ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q7_poly_C.png'))
    plt.show()
    plt.close()

def part_c_gamma(X, y):
    poly_svm = SVC(kernel='poly', C=100)
    cv_score_gamma_test = []
    cv_score_gamma_train = []
    # C_values = np.arange(0.01, 100, 0.01)
    gamma_values = [np.power(10.0,x) for x in np.arange(-7,7,1)]
    for g in gamma_values:
        print(g)
        poly_svm.set_params(gamma=g)
        scores = cross_validate(poly_svm, X, y, return_train_score=True, n_jobs=-1)
        print(scores['test_score'].mean())
        print(scores['train_score'].mean())
        cv_score_gamma_test.append(scores['test_score'].mean())
        cv_score_gamma_train.append(scores['train_score'].mean())

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax1.plot(gamma_values, cv_score_gamma_test, label='test score')
    ax1.plot(gamma_values, cv_score_gamma_train, label='train_score')


    ax1.set_xlabel('C')
    ax1.set_ylabel('Score')
    ax1.set_title('Model performance for gamma')
    ax1.legend(fontsize = 'large')
    ax1.set_xscale('log')

    plt.savefig(os.path.join(IMAGE_DIR, 'q7_poly_C.png'))
    plt.show()
    plt.close()

def main():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)
    auto_df = auto_df[auto_df['horsepower'] != '?']
    auto_df['horsepower'] = auto_df['horsepower'].apply(lambda x: float(x))

    print(auto_df.info())
    print(auto_df.isna().sum())

    y = auto_df['mpg01']
    X = auto_df.drop(['mpg01', 'name'], axis=1)

    # part_b(X, y)
    # part_c_Cvalue(X, y)
    # part_c_degree(X, y)
    part_c_gamma(X, y)

if __name__ == "__main__":
    main()

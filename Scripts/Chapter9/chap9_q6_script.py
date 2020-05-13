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

def part_a(x_1, x_2, y):
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    sns.scatterplot(x=x_1, y=x_2, hue=y, ax=ax)

    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_title('Data set with quasi-linear boundary with f')

    # plt.savefig(os.path.join(IMAGE_DIR,'q6_pa_class_plot.png'))
    plt.show()
    plt.close()

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

def main():
    x_1 = 10*np.random.uniform(size=500) - 5
    x_2 = 10*np.random.uniform(size=500) - 5
    X = np.column_stack((x_1, x_2))

    y = []

    for i in range(len(x_2)):
        arg = x_1[i] - x_2[i] + np.random.normal(scale=1.5, size=1)
        if arg > 0:
            y.append(1)
        else:
            y.append(-1)
    y = np.array(y)

    # part_a(x_1, x_2, y)
    part_b(X, y)

if __name__ == "__main__":
    main()

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

def part_b(x_1, x_2, y):
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    sns.scatterplot(x=x_1, y=x_2, hue=y, ax=ax)

    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_title('Data set with quadratic decision boundary')

    plt.savefig(os.path.join(IMAGE_DIR,'q5_pb_class_plot.png'))
    # plt.show()
    plt.close()

def part_cd(X, y):
    logr = LogisticRegression()
    logr.fit(X,y)
    y_preds = logr.predict(X)

    fig, ax = plt.subplots(1,1,figsize=(12,12))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_preds, ax =ax)

    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_title('Predicted class set with vanilla Logistic Regression')

    plt.savefig(os.path.join(IMAGE_DIR,'q5_pcd_class_plot.png'))
    # plt.show()
    plt.close()

def part_ef(X, y):
    ## Preprocessing
    x_12 = np.multiply(X[:,0], X[:,1]).reshape(X.shape[0], 1)
    # print(x_12)
    x_1_squared = np.multiply(X[:,0], X[:,0]).reshape(X.shape[0], 1)
    x_2_squared = np.multiply(X[:,1], X[:,1]).reshape(X.shape[0], 1)
    # x_1_log = np.log(X[:,0])
    # x_2_log = np.log(X[:,1])
    extra_pred_list = [x_12, x_1_squared, x_2_squared]

    for arr in extra_pred_list:
        X = np.append(X, arr, axis=1)

    logr = LogisticRegression()
    logr.fit(X,y)
    y_preds = logr.predict(X)

    fig, ax = plt.subplots(1,1,figsize=(12,12))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_preds, ax =ax)

    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_title('Predicted class set with feature expanded Logistic Regression')

    plt.savefig(os.path.join(IMAGE_DIR,'q5_pef_class_plot.png'))
    # plt.show()
    plt.close()

def part_g(X, y):
    lin_svc = SVC(kernel='linear', C=10)
    lin_svc.fit(X,y)
    y_preds = lin_svc.predict(X)

    fig, ax = plt.subplots(1,1,figsize=(12,12))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_preds, ax =ax)

    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_title('Predicted class set with linear SVC')

    plt.savefig(os.path.join(IMAGE_DIR,'q5_pg_class_plot.png'))
    # plt.show()
    plt.close()

def part_h(X, y):
    poly_svc = SVC(kernel='poly', C=10, degree=2)
    poly_svc.fit(X,y)
    y_preds = poly_svc.predict(X)

    fig, ax = plt.subplots(1,1,figsize=(12,12))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_preds, ax =ax)

    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_title('Predicted class set with quadratic SVC')

    plt.savefig(os.path.join(IMAGE_DIR,'q5_ph_class_plot.png'))
    # plt.show()
    plt.close()


def main():
    x_1 = np.random.uniform(size=500) - 0.5
    x_2 = np.random.uniform(size=500) - 0.5
    X = np.column_stack((x_1, x_2))

    y = []

    for i in range(len(x_2)):
        arg = x_1[i]**2 - x_2[i]**2
        if arg > 0:
            y.append(1)
        else:
            y.append(-1)
    y = np.array(y)

    part_b(x_1, x_2, y)
    part_cd(X, y)
    part_ef(X, y)
    part_g(X, y)
    part_h(X, y)


if __name__ == "__main__":
    main()

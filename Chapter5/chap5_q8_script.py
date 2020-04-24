import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter5')

def part_a():
    ## Generate the varaibles with normal distributions
    np.random.seed(100)
    x = np.random.randn(100)
    y = x - 2*x**2 + np.random.randn(100)
    return (x,y)

def part_b(x, y):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.scatterplot(x=x,y=y, ax=ax)
    plt.show()

def part_stand_err(x,y):
    width=10
    print("|{}|{}|{}|{}|{}|".format("Intercept".ljust(width),
                                    "X".ljust(width),
                                    "X^2".ljust(width),
                                    "X^3".ljust(width),
                                    "X^4".ljust(width)
                                    ))
    print("|{}|{}|{}|{}|{}|".format("-"*width,
                                    "-"*width,
                                    "-"*width,
                                    "-"*width,
                                    "-"*width
                                    ))
    ### part i
    X = x
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    # print(results.summary())
    print("|{}|{}|{}|{}|{}|".format(str(round(results.bse[0],3)).ljust(width),
                                    str(round(results.bse[1],3)).ljust(width),
                                    " ".ljust(width),
                                    " ".ljust(width),
                                    " ".ljust(width)
                                    ))

    ### part ii
    x = x.reshape(len(x), 1)
    x_squared = x*x

    X = np.concatenate((x, x_squared), axis=1)
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    # print(results.summary())
    print("|{}|{}|{}|{}|{}|".format(str(round(results.bse[0],3)).ljust(width),
                                    str(round(results.bse[1],3)).ljust(width),
                                    str(round(results.bse[2],3)).ljust(width),
                                    " ".ljust(width),
                                    " ".ljust(width)
                                    ))

    ### part iii
    x = x.reshape(len(x), 1)
    x_squared = x*x
    x_cubed = x**3

    X = np.concatenate((x, x_squared, x_cubed), axis=1)
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    # print(results.summary())
    print("|{}|{}|{}|{}|{}|".format(str(round(results.bse[0],3)).ljust(width),
                                    str(round(results.bse[1],3)).ljust(width),
                                    str(round(results.bse[2],3)).ljust(width),
                                    str(round(results.bse[3],4)).ljust(width),
                                    " ".ljust(width)
                                    ))

    ### part iv
    x = x.reshape(len(x), 1)
    x_squared = x*x
    x_cubed = x**3
    x_fourth = x**4

    X = np.concatenate((x, x_squared, x_cubed, x_fourth), axis=1)
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    # print(results.summary())
    print("|{}|{}|{}|{}|{}|".format(str(round(results.bse[0],3)).ljust(width),
                                    str(round(results.bse[1],3)).ljust(width),
                                    str(round(results.bse[2],3)).ljust(width),
                                    str(round(results.bse[3],4)).ljust(width),
                                    str(round(results.bse[4],3)).ljust(width)
                                    ))

    # for result in dir(results):
    #     print(result)

def part_c(x,y):
    ## Write own LOOCV
    length = len(x)
    errors = []
    for index in range(0,length -1):
        X = sm.add_constant(x)
        x_val_sample = X[index]
        x_val_sample = sm.add_constant(x_val_sample)
        # print(x_val_sample)
        y_val_sample= y[index]
        # print(y_val_sample)

        train_x = np.delete(x, index)
        # print(train_x)
        train_y = np.delete(y, index)

        # part i
        X = train_x
        X = sm.add_constant(X)
        model = sm.OLS(train_y, X)
        model.fit()
        prediction = model.predict(x_val_sample)
        print(prediction)
        mse = (prediction - y_val_sample)**2
        errors.append(mse)
    # print(errors)



def main():
    (x, y) = part_a()
    # print(x[0])
    # part_b(x,y)
    part_c(x,y)
    # part_d()


if __name__ == "__main__":
    main()

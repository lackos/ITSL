import pandas as pd
import numpy as np
import random

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter3')


def main():
    ###
    ### Question 11
    ###

    ### a) Generate a normal distribution of x values and related y values with normally distributed errors.
    random.seed(1)
    x = np.random.normal(size=100)
    # print(x)
    y = 2*x + np.random.normal(size=100)
    # print(y)
    ## Create a dataframe with these values
    data = {'y': y, 'x': x}
    df = pd.DataFrame(data=data)

    ## Perform the simple linear regression.
    results = smf.ols("y ~ x -1" , data=df).fit()
    print(results.summary())









if __name__ == "__main__":
    main()

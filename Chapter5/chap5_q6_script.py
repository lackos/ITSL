import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter5')

np.random.seed(100)

def part_a():
    # Load the default dataset
    default_df = pd.read_csv(os.path.join(DATA_DIR, 'default.csv'))
    binary_dict = {'No':0, 'Yes':1}
    default_df['default_bin'] = default_df['default'].map(binary_dict)

    # Fit the logistic regression (no train test split or cross val)
    X = default_df[['income', 'balance']]
    y = default_df['default_bin']

    ##Instantiate and fit the logistic regression with a single variable
    logit_reg = smf.logit(formula = "default_bin ~ income + balance", data= default_df).fit()
    print(logit_reg.summary())
    ## Standard Error: 0.435

    ## Predict the test data
    preds = logit_reg.predict(X)
    print(preds)

def main():
    part_a()

if __name__ == "__main__":
    main()

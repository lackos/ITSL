import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter5')

def part_a():
    # Load the default dataset
    default_df = pd.read_csv(os.path.join(DATA_DIR, 'default.csv'))

    # Fit the logistic regression (no train test split or cross val)
    X = default_df[['income', 'balance']]
    y = default_df['default']
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X, y)
    preds = log_reg_model.predict(X)

def main():
    part_a()

if __name__ == "__main__":
    main()

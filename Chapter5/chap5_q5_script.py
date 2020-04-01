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
    # print(default_df.head())

    # Fit the logistic regression (no train test split or cross val)
    X = default_df[['income', 'balance']]
    y = default_df['default']
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X, y)
    preds = log_reg_model.predict(X)

    # Print the classification report for the entire dataset
    print(classification_report(preds, y))

def part_b():
    ## Load the default dataset
    default_df = pd.read_csv(os.path.join(DATA_DIR, 'default.csv'))

    ## i) Manually split the data into training and validation (could also use train_test_split from sklearn)
    ### Shuffle the data
    # print(default_df.head())
    index = default_df.index
    default_df = shuffle(default_df)
    default_df.index = index
    # print(default_df.head())
    ## Set train set to 80% and validation to 20%
    size = default_df.shape[0]
    train = default_df.loc[0:0.8*size - 1]
    validation = default_df.loc[0.8*size:]
    # print(train)
    # print(validation)

    ## ii) Fit logistic regression to the training set
    X_train = train[['income', 'balance']]
    X_val = validation[['income', 'balance']]
    y_train = train['default']
    y_val = validation['default']
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)

    ## iii) Predict the default status of the validation set
    preds = log_reg_model.predict(X_val)
    ### Note: The question asks to set the posterior probability threshold to 0.5. SKlearn does this automatically with binary classification. To see how this is done using statsmodels refer to the solutions of Chapter 4.

    ## iv) Compute the test error
    cm = confusion_matrix(preds, y_val)
    test_error = (cm[0,1] + cm[1,0])/(len(preds))
    print('Test Error: ',test_error)
    return test_error
    ## As the random state is not set in the shuffle, rerunning this code gives different test errors. On repeated runs I found errors between 2-5% showing the variance in a single validation set split approach.

def part_c():
    errors = []
    for n in range(0,5):
        errors.append(part_b())

    errors = np.asarray(errors)
    print("Average Error: {}%".format(round(errors.mean()*100, 2)))

def part_d():
    default_df = pd.read_csv(os.path.join(DATA_DIR, 'default.csv'))
    ## Set dummy variables. Returns a binary column called 'student_Yes'
    default_df = pd.get_dummies(default_df, columns=['student'], drop_first=True)

    index = default_df.index
    default_df = shuffle(default_df)
    default_df.index = index
    size = default_df.shape[0]
    train = default_df.loc[0:0.8*size - 1]
    validation = default_df.loc[0.8*size:]


    ## Fit logistic regression to the training set
    X_train = train[['income', 'balance', 'student_Yes']]
    X_val = validation[['income', 'balance', 'student_Yes']]
    y_train = train['default']
    y_val = validation['default']
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)
    preds = log_reg_model.predict(X_val)

    ## Compute the test error
    cm = confusion_matrix(preds, y_val)
    test_error = (cm[0,1] + cm[1,0])/(len(preds))
    print('Test Error: ',test_error)
    return test_error
    ## Similar results with and without student


def main():
    # part_a()
    # part_b()
    # part_c()
    part_d()


if __name__ == "__main__":
    main()

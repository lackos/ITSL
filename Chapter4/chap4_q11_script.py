import pandas as pd
import numpy as np

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter4')

def part_a():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)
    print(auto_df.head())

def part_b():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)
    print(auto_df.info())
    print(auto_df.describe())
    print(auto_df.columns)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) =plt.subplots(nrows=3, ncols=2)
    sns.boxplot(data=auto_df, x='mpg01', y='cylinders', ax=ax1)
    sns.boxplot(data=auto_df, x='mpg01', y='displacement',ax=ax2)
    sns.boxplot(data=auto_df, x='mpg01', y='weight', ax=ax3)
    sns.boxplot(data=auto_df, x='mpg01', y='acceleration', ax=ax4)
    sns.boxplot(data=auto_df, x='mpg01', y='year', ax=ax5)
    sns.boxplot(data=auto_df, x='mpg01', y='origin', ax=ax6)
    plt.savefig(os.path.join(IMAGE_DIR, 'q11_boxplots.png'))
    # plt.show()
    plt.close()

    ## Scatter plot with mpg01
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) =plt.subplots(nrows=3, ncols=2, figsize=(15,15))
    sns.regplot(data=auto_df, y='mpg01', x='cylinders', ax=ax1, logistic=True)
    sns.regplot(data=auto_df, y='mpg01', x='displacement',ax=ax2, logistic=True)
    sns.regplot(data=auto_df, y='mpg01', x='weight', ax=ax3, logistic=True)
    sns.regplot(data=auto_df, y='mpg01', x='acceleration', ax=ax4, logistic=True)
    sns.regplot(data=auto_df, y='mpg01', x='year', ax=ax5, logistic=True)
    sns.regplot(data=auto_df, y='mpg01', x='origin', ax=ax6, logistic=True)
    plt.savefig(os.path.join(IMAGE_DIR, 'q11_scatterplots.png'))
    plt.tight_layout()
    # plt.show()
    plt.close()

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) =plt.subplots(nrows=3, ncols=2, figsize=(15,15))
    sns.distplot(auto_df['cylinders'], bins=30, ax=ax1)
    sns.distplot(auto_df['displacement'], bins=30, ax=ax2)
    sns.distplot(auto_df['weight'], bins=30, ax=ax3)
    sns.distplot(auto_df['acceleration'], bins=30, ax=ax4)
    sns.distplot(auto_df['year'], bins=30, ax=ax5)
    sns.distplot(auto_df['origin'], bins=30, ax=ax6)
    plt.savefig(os.path.join(IMAGE_DIR, 'q11_distplots.png'))
    plt.tight_layout()
    # plt.show()
    plt.close()

def part_c():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)
    print(auto_df.info())
    print(auto_df.describe())
    print(auto_df.columns)
    print(auto_df['mpg01'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(auto_df[['cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']], auto_df['mpg01'], test_size=0.2, random_state=1)
    print(y_train.value_counts())

def part_d():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)

    X_train, X_test, y_train, y_test = train_test_split(auto_df[['cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']], auto_df['mpg01'], test_size=0.2, random_state=1)

    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train[['cylinders', 'displacement', 'weight', 'acceleration']], y_train)
    preds = lda_model.predict(X_test[['cylinders', 'displacement', 'weight', 'acceleration']])
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    accuracy = round((tp + tn)/(len(preds)),2)
    print(cm)
    print('\nTrue Negatives(TN) = ', tn)

    print('\nTrue Positives(TP) = ', tp)

    print('\nFalse Negatives(FN) = ', fn)

    print('\nFalse Positives (FN) = ', fp)

    print('\nAccuracy = ', accuracy)

def part_e():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)

    X_train, X_test, y_train, y_test = train_test_split(auto_df[['cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']], auto_df['mpg01'], test_size=0.2, random_state=1)

    qda_model = QuadraticDiscriminantAnalysis()
    qda_model.fit(X_train[['cylinders', 'displacement', 'weight', 'acceleration']], y_train)
    preds = qda_model.predict(X_test[['cylinders', 'displacement', 'weight', 'acceleration']])
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    accuracy = round((tp + tn)/(len(preds)),2)
    print(cm)
    print('\nTrue Negatives(TN) = ', tn)

    print('\nTrue Positives(TP) = ', tp)

    print('\nFalse Negatives(FN) = ', fn)

    print('\nFalse Positives (FN) = ', fp)

    print('\nAccuracy = ', accuracy)

def classifier(value, threshold):
    if value >= threshold:
        return 1
    else:
        return 0

def part_f():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)

    X_train, X_test, y_train, y_test = train_test_split(auto_df[['cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']], auto_df['mpg01'], test_size=0.2, random_state=1)

    log_reg_results = smf.logit(formula = "mpg01 ~ cylinders + displacement + weight + acceleration", data= auto_df).fit()
    print(log_reg_results.summary())
    preds = log_reg_results.predict(X_test[['cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']])
    preds = preds.apply(lambda x: classifier(x, 0.5))
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    accuracy = round((tp + tn)/(len(preds)),2)
    print(cm)
    print('\nTrue Negatives(TN) = ', tn)

    print('\nTrue Positives(TP) = ', tp)

    print('\nFalse Negatives(FN) = ', fn)

    print('\nFalse Positives (FN) = ', fp)

    print('\nAccuracy = ', accuracy)

def part_g():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    median_mpg = auto_df['mpg'].describe()['50%']
    auto_df['mpg01'] = auto_df['mpg'].apply(lambda x: 1 if x > median_mpg else 0)

    X_train, X_test, y_train, y_test = train_test_split(auto_df[['cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']], auto_df['mpg01'], test_size=0.2, random_state=1)

    width=20
    print("N".ljust(width) + "|" + "True negatives".ljust(width) + "|" + "True Positives".ljust(width) + "|" + "False Negatives".ljust(width) + "|"+ "False Positives".ljust(width)+ "|" + "Accuracy".ljust(width))
    separator = "|---"
    print(separator*5 + '|')
    tn = {}
    fp = {}
    fn = {}
    tp = {}
    accuracy = {}
    for n in range(1,20):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train[['cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']], y_train)
        preds = knn.predict(X_test[['cylinders', 'displacement', 'weight', 'acceleration', 'year', 'origin']])
        cm = confusion_matrix(y_test, preds)
        tn[n], fp[n], fn[n], tp[n] = confusion_matrix(y_test, preds).ravel()
        accuracy[n] = round((tp[n] + tn[n])/(len(preds)),2)
        # print(cm)
        print("{}|{}|{}|{}|{}|{}|".format(str(n).ljust(width),str(tn[n]).ljust(width), str(tp[n]).ljust(width), str(fn[n]).ljust(width), str(fp[n]).ljust(width), str(accuracy[n]).ljust(width)))


def main():
    part_g()

if __name__ == "__main__":
    main()

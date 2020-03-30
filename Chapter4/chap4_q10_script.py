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

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter4')
def part_a():
    ## Load the 'Weekly' data set
    weekly_df = pd.read_csv(os.path.join(DATA_DIR, 'weekly.csv'))
    print(weekly_df.head())
    print(weekly_df.info())
    print(weekly_df.nunique())
    print(weekly_df['Direction'].value_counts())

    ## Make the response variable binary
    binary_dict = {'Down':0, 'Up':1}
    weekly_df['Direction_binary'] = weekly_df['Direction'].map(binary_dict)

    ## Print pairplot of the data
    sns.pairplot(data=weekly_df, diag_kind='kde', hue='Direction')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR,'q10_pairplot_hued.png'), dpi=500, pad_inches=0.2)
    plt.close()

    ## Log transform the volume column and plot the distribution
    weekly_df['log_volume'] = np.log(weekly_df['Volume'])
    sns.distplot(weekly_df['log_volume'], bins=50)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR,'q10_log_volume_dist.png'))
    plt.close()

    ## Boxplots of Direction against the numeric variables
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
    sns.boxplot(data=weekly_df, x='Direction', y='Lag1', ax=ax1)
    sns.boxplot(data=weekly_df, x='Direction', y='Lag2', ax=ax2)
    sns.boxplot(data=weekly_df, x='Direction', y='Lag3', ax=ax3)
    sns.boxplot(data=weekly_df, x='Direction', y='Lag4', ax=ax4)
    sns.boxplot(data=weekly_df, x='Direction', y='Lag5', ax=ax5)
    sns.boxplot(data=weekly_df, x='Direction', y='Volume', ax=ax6)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR,'q10_boxplots.png'))
    plt.close()

    ## Scatter plots with logistic fitting for Direction variable
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) =plt.subplots(nrows=3, ncols=2, figsize=(15,15))
    sns.regplot(data=weekly_df, y='Direction_binary', x='Lag1', ax=ax1, logistic=True)
    sns.regplot(data=weekly_df, y='Direction_binary', x='Lag2',ax=ax2, logistic=True)
    sns.regplot(data=weekly_df, y='Direction_binary', x='Lag3', ax=ax3, logistic=True)
    sns.regplot(data=weekly_df, y='Direction_binary', x='Lag4', ax=ax4, logistic=True)
    sns.regplot(data=weekly_df, y='Direction_binary', x='Lag5', ax=ax5, logistic=True)
    sns.regplot(data=weekly_df, y='Direction_binary', x='Volume', ax=ax6, logistic=True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'q10_scatterplots.png'))
    # plt.show()
    plt.close()

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) =plt.subplots(nrows=3, ncols=2, figsize=(15,15))
    sns.distplot(weekly_df['Lag1'], bins=30, ax=ax1)
    sns.distplot(weekly_df['Lag2'], bins=30, ax=ax2)
    sns.distplot(weekly_df['Lag3'], bins=30, ax=ax3)
    sns.distplot(weekly_df['Lag4'], bins=30, ax=ax4)
    sns.distplot(weekly_df['Lag5'], bins=30, ax=ax5)
    sns.distplot(weekly_df['Volume'], bins=30, ax=ax6)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'q10_distplots.png'))
    # plt.show()
    plt.close()

def part_b():
    weekly_df = pd.read_csv(os.path.join(DATA_DIR, 'weekly.csv'))
    ## Convert target variable into binary with simple map. 0-Down 1-Up
    binary_dict = {'Down':0, 'Up':1}
    weekly_df['Direction_binary'] = weekly_df['Direction'].map(binary_dict)
    ## Multivariable logistic regression with 'Direction' as target/response
    log_reg_results = smf.logit(formula = "Direction_binary ~ Volume +  Lag1 + Lag2 + Lag3 + Lag4 + Lag5", data= weekly_df).fit()
    print(log_reg_results.summary())
    ## Use the model to predict the values
    preds = log_reg_results.predict(weekly_df[['Volume', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5']])
    ## Set the threshold at 50% for classifying between Up or Down.
    preds = preds.apply(lambda x: classifier(x, 0.5))

def part_c():
    weekly_df = pd.read_csv(os.path.join(DATA_DIR, 'weekly.csv'))
    binary_dict = {'Down':0, 'Up':1}
    weekly_df['Direction_binary'] = weekly_df['Direction'].map(binary_dict)
    log_reg_results = smf.logit(formula = "Direction_binary ~ Volume +  Lag1 + Lag2 + Lag3 + Lag4 + Lag5", data= weekly_df).fit()
    preds = log_reg_results.predict(weekly_df[['Volume', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5']])
    preds = preds.apply(lambda x: classifier(x, 0.5))
    print(preds.value_counts())

    cm = confusion_matrix(weekly_df['Direction'], preds, labels=["Down", "Up"])
    tn, fp, fn, tp = confusion_matrix(weekly_df['Direction'], preds, labels=["Down", "Up"]).ravel()
    print(cm)
    sensitivity = round(cm[1,1]/(cm[1,1] + cm[0,1]), 2)
    specificity = round(1 - cm[1,0]/(cm[1,0] + cm[0,0]), 2)
    accuracy = round((tp + tn)/(len(preds)),2)
    cm_matrix = pd.DataFrame(data=cm, columns=['Predicted Down', 'Predicted Up'],
                                 index=['True Down', 'True Up'])

    print('\nTrue Negatives(TN) = ', tn)

    print('\nTrue Positives(TP) = ', tp)

    print('\nFalse Negatives(FN) = ', fn)

    print('\nFalse Positives (FN) = ', fp)

    print('\nSpecificity = ', specificity)

    print('\nSensitivity =', sensitivity)

    print('\nAccuracy = ', accuracy)
    hm = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR,'q10_log_reg_cm_0.5.png'))

def part_d():
    ## Load the data
    weekly_df = pd.read_csv(os.path.join(DATA_DIR, 'weekly.csv'))

    ## Make the response variable binary
    binary_dict = {'Down':0, 'Up':1}
    weekly_df['Direction_binary'] = weekly_df['Direction'].map(binary_dict)

    ## Break up into historical and current data
    train_data = weekly_df[(weekly_df['Year'] <= 2008)]
    test_data = weekly_df[(weekly_df['Year'] > 2008)]

    ##Instantiate and fit the logistic regression with a single variable
    logit_reg = smf.logit(formula = "Direction_binary ~ Lag2", data= train_data).fit()
    print(logit_reg.summary())

    ## Predict the test data
    preds = logit_reg.predict(test_data['Lag2'])

    ## Classify the result based on a 50% probability threshold and compute the confusion matrix
    preds = preds.apply(lambda x: classifier(x, 0.5))
    cm = confusion_matrix(test_data['Direction'], preds, labels=["Down", "Up"])
    tn, fp, fn, tp = confusion_matrix(test_data['Direction'], preds, labels=["Down", "Up"]).ravel()
    accuracy = round((tp + tn)/(len(preds)),2)

    ## Print the Results
    print(test_data['Direction'].value_counts())
    print(cm)
    print('\nTrue Negatives(TN) = ', tn)

    print('\nTrue Positives(TP) = ', tp)

    print('\nFalse Negatives(FN) = ', fn)

    print('\nFalse Positives (FN) = ', fp)

    print('\nAccuracy = ', accuracy)

def part_e():
    ## e) LDA method
    ## Load the data and break up into historical data
    weekly_df = pd.read_csv(os.path.join(DATA_DIR, 'weekly.csv'))
    train_data = weekly_df[(weekly_df['Year'] <= 2008)]
    test_data = weekly_df[(weekly_df['Year'] > 2008)]

    ## Reshape the data in the form of (len,1) arrays. sklearn will throw an error otherwise
    X_train = train_data['Lag2'].values.reshape(len(train_data), 1)
    y_train = train_data['Direction'].values.reshape(len(train_data), )
    X_test = test_data['Lag2'].values.reshape(len(test_data), 1)
    y_test = test_data['Direction'].values.reshape(len(test_data), )

    ## Instantiate the LDA model and fit it with the training data
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    ## Predict the values and generate the confusion matrix
    preds = lda.predict(X_test)
    cm = confusion_matrix(y_test, preds, labels=["Down", "Up"])
    tn, fp, fn, tp = confusion_matrix(test_data['Direction'], preds, labels=["Down", "Up"]).ravel()
    accuracy = round((tp + tn)/(len(preds)),2)

    print(test_data['Direction'].value_counts())
    print(cm)
    print('\nTrue Negatives(TN) = ', tn)

    print('\nTrue Positives(TP) = ', tp)

    print('\nFalse Negatives(FN) = ', fn)

    print('\nFalse Positives (FN) = ', fp)

    print('\nAccuracy = ', accuracy)

def part_f():
    ## QDA with one predictor
    ## Load the data and break up into historical data
    weekly_df = pd.read_csv(os.path.join(DATA_DIR, 'weekly.csv'))
    train_data = weekly_df[(weekly_df['Year'] <= 2008)]
    test_data = weekly_df[(weekly_df['Year'] > 2008)]

    ## Reshape the data in the form of (len,1) arrays. sklearn will throw an error otherwise
    X_train = train_data['Lag2'].values.reshape(len(train_data), 1)
    y_train = train_data['Direction'].values.reshape(len(train_data), )
    X_test = test_data['Lag2'].values.reshape(len(test_data), 1)
    y_test = test_data['Direction'].values.reshape(len(test_data), )

    ## Instantiate the LDA model and fit it with the training data
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)

    ## Predict the values and generate the confusion matrix
    preds = qda.predict(X_test)
    print(preds)
    cm = confusion_matrix(y_test, preds, labels=["Down", "Up"])
    tn, fp, fn, tp = confusion_matrix(test_data['Direction'], preds, labels=["Down", "Up"]).ravel()
    accuracy = round((tp + tn)/(len(preds)),2)

    print(test_data['Direction'].value_counts())
    print(cm)
    print('\nTrue Negatives(TN) = ', tn)

    print('\nTrue Positives(TP) = ', tp)

    print('\nFalse Negatives(FN) = ', fn)

    print('\nFalse Positives (FN) = ', fp)

    print('\nAccuracy = ', accuracy)

def part_g():
    ### KNN with one predictor
    weekly_df = pd.read_csv(os.path.join(DATA_DIR, 'weekly.csv'))

    train_data = weekly_df[(weekly_df['Year'] <= 2008)]
    test_data = weekly_df[(weekly_df['Year'] > 2008)]

    X_train = train_data['Lag2'].values.reshape(len(train_data), 1)
    y_train = train_data['Direction'].values.reshape(len(train_data), )
    X_test = test_data['Lag2'].values.reshape(len(test_data), 1)
    y_test = test_data['Direction'].values.reshape(len(test_data), )

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    cm = confusion_matrix(y_test, preds, labels=["Down", "Up"])
    tn, fp, fn, tp = confusion_matrix(test_data['Direction'], preds, labels=["Down", "Up"]).ravel()
    accuracy = round((tp + tn)/(len(preds)),2)
    print(test_data['Direction'].value_counts())
    print(cm)
    print('\nTrue Negatives(TN) = ', tn)

    print('\nTrue Positives(TP) = ', tp)

    print('\nFalse Negatives(FN) = ', fn)

    print('\nFalse Positives (FN) = ', fp)

    print('\nAccuracy = ', accuracy)


def classifier(value, threshold):
    if value >= threshold:
        return 'Up'
    else:
        return 'Down'

def main():
    part_g()






if __name__ == "__main__":
    main()

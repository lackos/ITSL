import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter2')

def main():
    ###
    ### Question 9
    ###

    ## Import auto.csv from DATA_DIR
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))

    ## Print out missing values
    print(auto_df.isna().sum())
    print(auto_df.describe())


    ### a) Print out qualitative and quantitative columns
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    object_types = ['object']
    quantitative = auto_df.select_dtypes(include=numeric_types).columns
    qualitative = auto_df.select_dtypes(include=object_types).columns
    print("All features : " + str(auto_df.columns.format()))
    print("Quantitative features: " + str(quantitative.format()))
    print("Qualitative features: " + str(qualitative.format()))

    ### b) Print out the range of the quantitative features
    for feature in quantitative:
        print(feature + ' range: ' + str(auto_df[feature].describe()['min']) + ' to ' +              str(auto_df[feature].describe()['max']))

    ### c) Print the mean and standard dev of all the quantitative features
    for feature in quantitative:
        print(feature + ' mean: ' + str(round(auto_df[feature].describe()['mean'], 2)) + ' , std. dev.: ' + str(round(auto_df[feature].describe()['std'], 2)))

    print('\n\n')


    ### d) remove the 10th through 85 columns
    auto_dropped = auto_df.drop(auto_df.index[10:86])
    # print(auto_dropped)
    for feature in quantitative:
        print(feature + ' range: ' + str(auto_dropped[feature].describe()['min']) + ' to ' +              str(auto_dropped[feature].describe()['max']))
    for feature in quantitative:
        print(feature + ' mean: ' + str(round(auto_dropped[feature].describe()['mean'], 2)) + ' , std. dev.: ' + str(round(auto_dropped[feature].describe()['std'], 2)))

    ### e) predictors investigation
    ## Print scatter matrix
    ax1 = sns.pairplot(auto_df[quantitative])
    plt.gcf().subplots_adjust(bottom=0.05, left=0.1, top=0.95, right=0.95)
    ax1.fig.suptitle('Auto Quantitative Scatter Matrix', fontsize=35)
    # plt.savefig(os.path.join(IMAGE_DIR,'auto_scatter_matrix.png'), format='png', dpi=250)
    # plt.show()
    ## From the scatter plots we see that none of the predictors are normally distributed.
    ## Also there is a clear linear relationship between weight and displacement and
    ## a decay relationship between the mpg and weight which is to be expected. In general
    ## mpg increases as year increases.

    









if __name__ == "__main__":
    main()

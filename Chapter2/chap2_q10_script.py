import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.datasets import load_boston

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter2')

def main():
    ###
    ### Question 10
    ###
    ## Load boston dataset and save a csv in DATA_DIR
    # boston_data = load_boston()
    # df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
    # df_boston['target'] = pd.Series(boston_data.target)
    # df_boston.head()
    # print(df_boston.head())
    # df_boston.to_csv(os.path.join(DATA_DIR, 'boston.csv'), index=False)

    ## Load the boston dataset
    boston_df = pd.read_csv(os.path.join(DATA_DIR, 'boston.csv'))

    ## a) Dataset information
    print(boston_df.describe())
    print(len(boston_df.columns))
    print(boston_df.columns)
    ## There are 506 rows of data and 14 columns

    ## b) scatterplot matrix
    # ax1 = sns.pairplot(boston_df)
    # plt.gcf().subplots_adjust(bottom=0.05, left=0.1, top=0.95, right=0.95)
    # ax1.fig.suptitle('Boston Scatter Matrix', fontsize=35)
    # plt.savefig(os.path.join(IMAGE_DIR,'boston_scatter_matrix.png'), format='png', dpi=250)
    # plt.show()

    ## c) 

if __name__ == "__main__":
    main()

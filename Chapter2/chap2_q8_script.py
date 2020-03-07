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
    ### Question 8
    ###


    ### a)
    ## Import College.csv from DATA_DIR
    college_df = pd.read_csv(os.path.join(DATA_DIR, 'college.csv'))


    ### b)
    ## Print the first 5 rows of the dataframe
    # print(college_df.head())
    # print(college_df.columns)

    ## Set the column index to the College name
    college_df.rename(columns={college_df.columns[0] : "College"}, inplace=True)
    college_df.set_index("College", inplace=True)
    # print(college_df.head())

    ### c)
    ## Print the Summary of the dataframe
    # print(college_df.describe())

    ## Plot the scatter matrix of the first 11 numerical columns
    ax1 = sns.pairplot(college_df[college_df.columns[0:11]])
    plt.gcf().subplots_adjust(bottom=0.05, left=0.1, top=0.95, right=0.95)
    ax1.fig.suptitle('College Scatter Matrix', fontsize=35)

    ## Show/save the plot
    # plt.savefig(os.path.join(IMAGE_DIR,'college_scatter_matrix.png'), format='png', dpi=250)
    # plt.show()
    plt.close()

    ## Boxplot both Outstate and Private. I assume this means it wants boxplots of the Private Categories 'Yes' and 'No'
    ## and the number of Outstate
    fig, ax = plt.subplots()
    sns.boxplot(ax=ax, x="Private", y="Outstate", data=college_df)
    plt.title(r'Outstate vs Private Boxplots')
    width = 3.403
    height = 1.518*width
    fig.set_size_inches(width, height)
    fig.tight_layout()
    # plt.savefig(os.path.join(IMAGE_DIR,'Outstate_Private_boxplot.png'), format='png', dpi=250)
    # plt.show()
    plt.close()

    ## Bin the top universities into a new column 'Elite' and count number of elite universities
    college_df['Elite'] = pd.Series(len(college_df['Top10perc']), index=college_df.index)
    college_df['Elite'] = 'No'
    college_df.loc[college_df['Top10perc']>50,'Elite'] = 'Yes'
    num_elite = college_df[college_df['Elite'] == 'Yes']['Elite'].count()
    print("Number of Elite Universities: " + str(num_elite))

    ## Boxplot Elite vs Outstate
    fig, ax = plt.subplots()
    sns.boxplot(ax=ax, x="Elite", y="Outstate", data=college_df)
    plt.title(r'Outstate vs Elite Boxplots')
    width = 3.403
    height = 1.518*width
    fig.set_size_inches(width, height)
    fig.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR,'Outstate_Elite_boxplot.png'), format='png', dpi=250)
    plt.show()
    plt.close()



if __name__ == "__main__":
    main()

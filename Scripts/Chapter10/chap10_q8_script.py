import pandas as pd
import numpy as np
import random
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter10')

np.random.seed(102)
plt.style.use("seaborn-darkgrid")

def part_a(us_crime):
    ## Instantiate the scaler and fit the data to the columns
    scaler = StandardScaler()
    scaler.fit(us_crime[['Murder',  'Assault',  'UrbanPop',  'Rape']])
    X = scaler.transform(us_crime)

    ## Instantiate and fit the PCA model
    pca=PCA(n_components=4)
    pca.fit(X)
    pve_ratios = pca.explained_variance_ratio_

    cul_pve = [0]
    for idx, val in enumerate(pve_ratios):
        next = cul_pve[idx] + val
        cul_pve.append(next)
    cul_pve.pop(0)
    print(cul_pve)

    fig, ((ax1), (ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(24,12))

    g = sns.lineplot([0,1,2,3], pve_ratios,  ax=ax1)
    f = sns.lineplot([0,1,2,3], cul_pve,  ax=ax2)

    g.set_xticks([0,1,2,3])
    f.set_xticks([0,1,2,3])
    PCA_labels = ["PC1", "PC2", "PC3", "PC4"]
    g.set_xticklabels(PCA_labels)
    f.set_xticklabels(PCA_labels)



    ax1.set_xlabel("Principal components", fontsize='x-large')
    ax1.set_ylabel("Porportion of explained Variance", fontsize='x-large')

    ax2.set_xlabel("Principal components", fontsize='x-large')
    ax2.set_ylabel("Cumulative porportion of explained Variance", fontsize='x-large')

    ax1.set_title("Skree Plot", fontsize='xx-large')
    ax2.set_title("Cumulative Plot", fontsize='xx-large')

    plt.savefig(os.path.join(IMAGE_DIR, "p8_pve_plots.png"), dpi=500)
    plt.show()


def part_b(us_crime):
    ## Instantiate the scaler and fit the data to the columns
    scaler = StandardScaler()
    scaler.fit(us_crime[['Murder',  'Assault',  'UrbanPop',  'Rape']])
    X = scaler.transform(us_crime)

    pca=PCA(n_components=4)
    pca.fit(X)

    pca_mat = pca.components_

    ## For the first component only, m = 1
    ve = 0
    for i in range(X.shape[0]):
        inner_sum = 0
        ## For each observation calculate the linear combination of pca and value
        for j in range(pca_mat.shape[0]):
            iter = X[i,j]*pca_mat[0,j]
            inner_sum += iter
        ## Square this sum and add it to the total ve
        ve += inner_sum**2
    ## Divide by the number of observations
    ve = ve/X.shape[0]
    print("Variance explained for PC1: ", ve)

    ## Calculate the total variance of the data set
    tot_var = 0
    for i in range(X.shape[0]):
        inner_sum = 0
        ## squar the matrix element
        for j in range(X.shape[1]):
            iter = X[i,j]*X[i,j]
            inner_sum += iter
        ## Add the inner sum to the total variance
        tot_var += inner_sum
    tot_var = tot_var/X.shape[0]
    print("Total variance of dataset: ", tot_var)

    ## Calculate porportion of variance explained
    for m in range(pca_mat.shape[0]):
        pve = 0
        for i in range(X.shape[0]):
            inner_sum = 0
            ## For each observation calculate the linear combination of pca and value
            for j in range(pca_mat.shape[0]):
                iter = X[i,j]*pca_mat[m,j]
                inner_sum += iter
            ## Square this sum and add it to the total ve
            pve += inner_sum**2
        ## Divide by the number of observations and the total variance
        pve = pve/(X.shape[0]*tot_var)
        print("Porportion of variance explained by PC{0}: {1}".format(m+1, pve))

    print(pca.explained_variance_ratio_)


def main():
    ## Load the dataset
    us_crime = pd.read_csv(os.path.join(DATA_DIR, "USArrests.csv"), index_col='Unnamed: 0')

    # part_a(us_crime)
    part_b(us_crime)

if __name__ == "__main__":
    main()

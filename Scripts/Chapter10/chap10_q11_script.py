import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter10')

np.random.seed(102)

plt.style.use("seaborn-darkgrid")


def main():
    ## Load the dataset
    gene_df = pd.read_csv(os.path.join(DATA_DIR, 'ch10Ex11.csv'), names=list(range(0,40)), header=None)

    ## Create the correlation matrix
    gene_corr = gene_df.transpose().corr()
    print(gene_corr.shape)

    ## Convert correlation to dissimalarity
    ### Vectorize the function
    diss = np.vectorize(lambda x: 1- np.abs(x))
    diss_corr = gene_corr.apply(diss)

    print(diss_corr)

    gene_clusterer = AgglomerativeClustering(affinity=diss_corr.values, compute_full_tree='auto', linkage='complete', distance_threshold=0, n_clusters=None)

    gene_clusterer = gene_clusterer.fit(gene_df)

    plt.title('Complete Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(gene_clusterer, truncate_mode='level')
    plt.xlabel("gene")
    plt.show()


if __name__ == "__main__":
    main()

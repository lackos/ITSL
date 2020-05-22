import pandas as pd
import numpy as np
import random
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter10')

np.random.seed(102)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def euclid_cluster(data):
    ## Instantiate the scaler and fit the data to the columns
    scaler = StandardScaler()
    scaler.fit(data[['Murder',  'Assault',  'UrbanPop',  'Rape']])
    data = scaler.transform(data)

    heir_clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=0, n_clusters=None)

    heir_clusterer = heir_clusterer.fit(data)

    plt.title('Hierarchical Clustering Dendrogram (euclidean dissimilarity)')
    # plot the top three levels of the dendrogram
    plot_dendrogram(heir_clusterer, truncate_mode='level')
    plt.xlabel("States")
    plt.savefig(os.path.join(IMAGE_DIR, "p7_Euclid_dendogram.png"), dpi=500)
    plt.show()

def correlation_cluster(data):
    corr_mat = data.transpose().corr()
    diss = np.vectorize(lambda x: 1- np.abs(x), otypes=[np.float])
    diss_corr = corr_mat.apply(diss)

    fig, ((ax1), (ax2))  = plt.subplots(nrows=1, ncols=2, figsize=(24,12))
    sns.heatmap(corr_mat, ax=ax1)
    sns.heatmap(diss_corr, ax=ax2)

    ax1.set_title("Raw correlation matrix")
    ax2.set_title("Distance correlation matrix")
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, "p7_corr_dist_mat.png"), dpi=500)
    plt.close()

    ## Instantiate the scaler and fit the data to the columns
    scaler = StandardScaler()
    scaler.fit(data[['Murder',  'Assault',  'UrbanPop',  'Rape']])
    X = scaler.transform(data)
    corr_mat = np.corrcoef(X)
    diss_corr = diss(corr_mat)
    heir_clusterer = AgglomerativeClustering(affinity='precomputed', compute_full_tree='auto', linkage='complete', distance_threshold=0, n_clusters=None)

    heir_clusterer = heir_clusterer.fit(diss_corr)

    plt.title('Hierarchical Clustering Dendrogram (correlation dissimilarity)')
    # plot the top three levels of the dendrogram
    plot_dendrogram(heir_clusterer, truncate_mode='level')
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(os.path.join(IMAGE_DIR, "p7_correlation_dendogram.png"), dpi=500)
    # plt.show()


def main():
    ## Load dataset
    us_crime = pd.read_csv(os.path.join(DATA_DIR, "USArrests.csv"), index_col='Unnamed: 0')

    # euclid_cluster(us_crime)
    correlation_cluster(us_crime)




if __name__ == "__main__":
    main()

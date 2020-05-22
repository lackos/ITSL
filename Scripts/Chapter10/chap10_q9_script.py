import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter10')

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    # print(model.labels_)
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
    # print(linkage_matrix)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def part_a(data):
    heir_clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=0, n_clusters=None)

    heir_clusterer = heir_clusterer.fit(data)

    plt.title('Complete Hierarchical Clustering Dendrogram (No scaling)')
    # plot the top three levels of the dendrogram
    plot_dendrogram(heir_clusterer, truncate_mode='level')
    plt.xlabel("State")
    # plt.show()
    plt.savefig(os.path.join(IMAGE_DIR, "problem9_parta_dendogram.png"), dpi=500)

def part_b(data, index_dict):
    scaler = StandardScaler()
    scaler.fit(data[['Murder',  'Assault',  'UrbanPop',  'Rape']])
    data =  scaler.transform(data)
    heir_clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=None, n_clusters=3)

    heir_clusterer = heir_clusterer.fit(data)

    cluster_labels = {i: np.where(heir_clusterer.labels_ == i)[0] for i in range(heir_clusterer.n_clusters)}

    for c, state in cluster_labels.items():
        state_clusters = [index_dict[st] for st in state]
        print("Group ", c)
        print("--------")
        for st in state_clusters:
            print(st)
        print('\n')

def part_c(data, index_dict):
    heir_clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=None, n_clusters=3)

    heir_clusterer = heir_clusterer.fit(data)

    cluster_labels = {i: np.where(heir_clusterer.labels_ == i)[0] for i in range(heir_clusterer.n_clusters)}

    for c, state in cluster_labels.items():
        state_clusters = [index_dict[st] for st in state]
        print("Group ", c)
        print("--------")
        for st in state_clusters:
            print(st)
        print('\n')

def index_dict(pd_index):
    dict = {}
    for idx, val in enumerate(pd_index):
        dict[idx] = val
    return dict

def main():
    ## Load dataset
    us_crime = pd.read_csv(os.path.join(DATA_DIR, "USArrests.csv"), index_col='Unnamed: 0')
    idx_dict = index_dict(us_crime.index)

    part_a(us_crime)
    # part_b(us_crime, idx_dict)
    # part_c(us_crime, idx_dict)

if __name__ == "__main__":
    main()

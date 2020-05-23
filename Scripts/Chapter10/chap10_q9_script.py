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

    heir_clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=None, n_clusters=3)

    heir_clusterer = heir_clusterer.fit(data)

    cluster_labels = {i: np.where(heir_clusterer.labels_ == i)[0] for i in range(heir_clusterer.n_clusters)}

    state_dict = {}
    for c, state in cluster_labels.items():
        state_clusters = [index_dict[st] for st in state]
        state_dict[c] = state_clusters
        print("Group ", c)
        print("--------")
        for st in state_clusters:
            print(st)
        print('\n')
    return state_dict

def part_c(data, index_dict):
    scaler = StandardScaler()
    scaler.fit(data[['Murder',  'Assault',  'UrbanPop',  'Rape']])
    data =  scaler.transform(data)
    heir_clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=None, n_clusters=3)

    heir_clusterer = heir_clusterer.fit(data)

    cluster_labels = {i: np.where(heir_clusterer.labels_ == i)[0] for i in range(heir_clusterer.n_clusters)}

    state_dict = {}
    for c, state in cluster_labels.items():
        state_clusters = [index_dict[st] for st in state]
        state_dict[c] = state_clusters
        print("Group ", c)
        print("--------")
        for st in state_clusters:
            print(st)
        print('\n')
    return state_dict

def index_dict(pd_index):
    dict = {}
    for idx, val in enumerate(pd_index):
        dict[idx] = val
    return dict

def generate_table(us_crime, idx_dict):
    non_scaled = part_b(us_crime, idx_dict)
    scaled = part_c(us_crime, idx_dict)

    print(non_scaled)
    print(scaled)


    ## Print most common clusters
    sim_dict = {}
    for key1, val1 in non_scaled.items():
        sim_count = 0
        for key2, val2 in scaled.items():
            count = len(list(set(val1).intersection(val2)))
            if count > sim_count and key2 not in sim_dict.values():
                sim_count = count
                sim_dict[key1] = key2

    print(sim_dict)

    print("{} | {} ".format("Non-scaled", "Scaled"))
    print("---|----")
    for c, list1 in non_scaled.items():
        print("**Group {}** | **Group {}** ".format(c + 1, c + 1))
        list2 = []
        list2 = scaled[sim_dict[c]]
        for i in range(max(len(list1), len(list2))):
            if i >= len(list1):
                item1 = " "
                item2 = list2[i]
            elif i >= len(list2):
                item1 = list1[i]
                item2 = " "
            else:
                item1 = list1[i]
                item2 = list2[i]
            print("{} | {}".format(item1, item2))



def main():
    ## Load dataset
    us_crime = pd.read_csv(os.path.join(DATA_DIR, "USArrests.csv"), index_col='Unnamed: 0')
    idx_dict = index_dict(us_crime.index)

    # part_a(us_crime)
    # non_scaled = part_b(us_crime, idx_dict)
    # scaled = part_c(us_crime, idx_dict)
    generate_table(us_crime, idx_dict)

if __name__ == "__main__":
    main()

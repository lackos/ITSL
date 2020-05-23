import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter10')

np.random.seed(102)

plt.style.use("seaborn-darkgrid")

def part_b(data):
    ## Break the data into Principal components
    pca = PCA(n_components=2)
    pca.fit(data)

    X = pca.transform(data)
    print(X)

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    sns.scatterplot(x=X[0:20,0], y=X[0:20,1], color='r', label='Cluster 1')
    sns.scatterplot(x=X[20:40,0], y=X[20:40,1], color='b', label="Cluster 2")
    sns.scatterplot(x=X[40:60,0], y=X[40:60,1], color='g', label="Cluster 3")

    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")

    ax1.set_title("Principal component plot of three distinct clusters")

    ax1.legend()
    plt.savefig(os.path.join(IMAGE_DIR, "p10_partb_PCA.png"), dpi=500)
    plt.show()

    pass

def part_c(data):
    ## Break the data into Principal components
    kmeans = KMeans(n_clusters=3)

    X = kmeans.fit_predict(data)
    print("True Cluster 0 prdictions:")
    print(X[0:20])
    print("True Cluster 1 prdictions:")
    print(X[20:40])
    print("True Cluster 2 prdictions:")
    print(X[40:60])

    pass

def part_d(data):
    kmeans = KMeans(n_clusters=2)

    X = kmeans.fit_predict(data)
    print("True Cluster 0 prdictions:")
    print(X[0:20])
    print("True Cluster 1 prdictions:")
    print(X[20:40])
    print("True Cluster 2 prdictions:")
    print(X[40:60])

def part_e(data):
    kmeans = KMeans(n_clusters=4)

    X = kmeans.fit_predict(data)
    print("True Cluster 0 prdictions:")
    print(X[0:20])
    print("True Cluster 1 prdictions:")
    print(X[20:40])
    print("True Cluster 2 prdictions:")
    print(X[40:60])

def part_f(data):
    ## Break the data into Principal components
    pca = PCA(n_components=2)
    pca.fit(data)
    X_PCA = pca.transform(data)

    kmeans = KMeans(n_clusters=3)
    X = kmeans.fit_predict(X_PCA)
    print("True Cluster 0 prdictions:")
    print(X[0:20])
    print("True Cluster 1 prdictions:")
    print(X[20:40])
    print("True Cluster 2 prdictions:")
    print(X[40:60])

def part_g(data):
    ## Break the data into Principal components
    scaler = StandardScaler(with_mean=False)
    scaler.fit(data)
    data =  scaler.transform(data)

    kmeans = KMeans(n_clusters=3)

    X = kmeans.fit_predict(data)
    print("True Cluster 0 prdictions:")
    print(X[0:20])
    print("True Cluster 1 prdictions:")
    print(X[20:40])
    print("True Cluster 2 prdictions:")
    print(X[40:60])


def main():
    ## Generate the data
    true_cluster_1 = np.random.normal(loc=-5, scale=0.5, size=(20,50))
    true_cluster_2 = np.random.normal(loc=10, scale=1, size=(20,50))
    true_cluster_3 = np.random.normal(loc=5, scale=0.9, size=(20,50))

    ## Combine the data into a single array
    combined = np.append(true_cluster_1, np.append(true_cluster_2, true_cluster_3, axis=0), axis=0)

    # part_b(combined)
    # part_c(combined)
    # part_d(combined)
    # part_e(combined)
    # part_f(combined)
    part_g(combined)

if __name__ == "__main__":
    main()

# Chapter 10: Unsupervised Learning
# Applied Problems
Load the standard libraries

```python
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(102)
```

## Problem Seven
Load the US arrests dataset

```python
us_crime = pd.read_csv(os.path.join(DATA_DIR, "USArrests.csv"), index_col='Unnamed: 0')
```

Import and instantiate the standard scaler for scaling scaling th data such that $\mu = 0$ and $\sigma = 1$. Then fit and transform (can be done in a single step) the data.

Note: This will return a numpy array and we will therefore lose the state indices.

```python
from sklearn.preprocessing import StandardScaler

## Instantiate and fit the scaler
scaler = StandardScaler()
scaler.fit(us_crime[['Murder',  'Assault',  'UrbanPop',  'Rape']])
## Transform the data
us_crime =  scaler.transform(us_crime)
```

### Euclidean dissimalarity
First we will create a dendogram using a euclidean metric for the dissimilarity matrix. This is a distance matrix whose entries represent the distance in feature space between the $i$th and $j$th states for $p$ features,

\[
d_{ij} = \sqrt{\sum_{k=1}^p (x_{i,k} - x_{j,k})^2}.
\]

We instantiate and fit a hierarchial clustering (agglomerative) model. We use a 'complete' linkage model and compute the entire dendogram tree.

```python
from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=0, n_clusters=None)
```

To plot the dendogram we use the following function found in the documentation,

```python
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
```

The final euclidean dendogram is,

```python
clusterer = clusterer.fit(us_crime)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(heir_clusterer, truncate_mode='level')
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig(os.path.join(IMAGE_DIR, "p7_Euclid_dendogram.png"), dpi=500)
plt.show()
```

<img src="../../Images/Chapter10/p7_Euclid_dendogram.png" alt="bootstrap_limit" title="bootstrap_limit"  />

**To update: restore state names rather than indices to dendogram**.

### Correlation dissimilarity

For the correlation dissimilarity we need to construct our own dissimilarity matrix as it is not standard. As euclidean distance matrix relates to the $l2$ distance between observations, the correlation matrix denotes how correlated to observations are in the feature space. We will be using the 'Pearson' correlation. However, Pearson's correlation coefficient is 0 for no correlation, 1 for total positive correlation and -1 for total negative correlation. This is nearly the opposite of a distance matrix (0 for very similar and 1 for no similarity) and therefore we must transform each correlation value $r_{ij}$ with,

\[
d_{ij} = 1 - \left|r_{ij}\right|
\]

One thing to keep in mind here is the correlation target. In earlier sections we used the correlation matrix to determine the correlation of the fetures over the rows (observations) to help determine the relationship between the features (for examples see Chapter 3). Here we are doing the opposite, we want to find which observations correlate most highly with each other over the feature space. Therefore we will have to take the transpose of the dataframe.

To compare the difference between the raw correlation matrix and the correlation dissimilary matrix we have the following plot.


```python
## Transpose the data matrix and find correlation of rows (observations)
corr_mat = data.transpose().corr()
## Vectorize the correlation dissimilarity function
diss = np.vectorize(lambda x: 1- np.abs(x), otypes=[np.float])
## Generate a dataframe of the correlation dissimilarity
diss_corr = corr_mat.apply(diss)

fig, ((ax1), (ax2))  = plt.subplots(nrows=1, ncols=2, figsize=(24,12))
sns.heatmap(corr_mat, ax=ax1)
sns.heatmap(diss_corr, ax=ax2)
ax1.set_title("Raw correlation matrix")
ax2.set_title("Distance correlation matrix")
plt.show()
```

<img src="../../Images/Chapter10/p7_corr_dist_mat.png" alt="bootstrap_limit" title="bootstrap_limit"  />

To find the dendogram with the correlation dissimilarity we have to scale the features to the standard normal (it will still work without scaling but the denogram will be very squashed, Try it!)

```python
## Instantiate the scaler and fit the data to the columns
scaler = StandardScaler()
scaler.fit(data[['Murder',  'Assault',  'UrbanPop',  'Rape']])
X = scaler.transform(data)

## Find the correlation matrix and map the distance function
corr_mat = np.corrcoef(X)
diss_corr = diss(corr_mat)

## Instantiate the clusterer with 'precomputed' affinity.
heir_clusterer = AgglomerativeClustering(affinity='precomputed', compute_full_tree='auto', linkage='complete', distance_threshold=0, n_clusters=None)

## Fit the clustered to the pre computed distance matrix (correlation dissimilarity)
heir_clusterer = heir_clusterer.fit(diss_corr)

## Plot the resulting dendogram
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(heir_clusterer, truncate_mode='level')
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig(os.path.join(IMAGE_DIR, "p7_correlation_dendogram.png"), dpi=500)
plt.show()
```

<img src="../../Images/Chapter10/p7_correlation_dendogram.png" alt="bootstrap_limit" title="bootstrap_limit"  />

This does not appear to be similar to the euclidean distance matrix.



**Return to this**

## Problem Eight
### Part a)
This problem is simple using the predefined attributes of the PCA object in sklearn. We instantiate the PCA object with the following,
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## Load the dataset
us_crime = pd.read_csv(os.path.join(DATA_DIR, "USArrests.csv"), index_col='Unnamed: 0')
## Instantiate the scaler and fit the data to the columns
scaler = StandardScaler()
scaler.fit(us_crime[['Murder',  'Assault',  'UrbanPop',  'Rape']])
X = scaler.transform(us_crime)

## Instantiate and fit the PCA model
pca=PCA(n_components=4)
pca.fit(X)
```

The porportion of variance explained is then easily called with the attribute,
```python
pve_ratios = pca.explained_variance_ratio_
print(pve_ratios)
```

```
[0.62006039 0.24744129 0.0891408  0.04335752]
```

A cumulative set of PVE can be generated with

```python
cul_pve = [0]
for idx, val in enumerate(pve_ratios):
    next = cul_pve[idx] + val
    cul_pve.append(next)
cul_pve.pop(0)

print(cul_pve)
```

```
[0.62006039, 0.86750168, 0.95664248, 0.99999999]
```

We plot both the individual and cumulative PVEs below,
```python
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

plt.show()
```

<img src="../../Images/Chapter10/p8_pve_plots.png" alt="bootstrap_limit" title="bootstrap_limit"  />

### Part b)
Here we will calculate the pve's ourselves with the raw PCA matrix and dataset.

First we do it for a single PCA component (PC1).

*Note: The PCA matrix produced by sklearn is the transpose of the one described in the text. That is, we use $\phi_{mj}$, where $m$ are the PCs and $j$ are the original features.*

```python
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
```

```
Variance explained for PC1:  2.480241579149494
```

This can simply be calculated for every PC. To calculate the PVE we first calculate the total variance of the dataset,

```python
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
```

```
Total variance of dataset:  4.0
```

We can now calculate the porportion of variance explained for each  PC.


```python
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
```

```
Porportion of variance explained by PC1: 0.6200603947873735
Porportion of variance explained by PC2: 0.24744128813496036
Porportion of variance explained by PC3: 0.08914079514520756
Porportion of variance explained by PC4: 0.043357521932458856
```

## Problem Nine
### Part a)
Load the dataset
```python
## Load dataset
data = pd.read_csv(os.path.join(DATA_DIR, "USArrests.csv"), index_col='Unnamed: 0')
```
As transforming the data removes the indicies (state name) we define a function to create dictionary of the index location and original index.
```python
## Create a dictionary of the index and state name for restoring the index after transformations
def index_dict(pd_index):
    dict = {}
    for idx, val in enumerate(pd_index):
        dict[idx] = val
    return dict
```

Index the dataframe.
```python
idx_dict = index_dict(us_crime.index)
```
Now fit the hierarchial clusterer with euclidean dissimilarity and complete linkage
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

hier_clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=0, n_clusters=None)

hier_clusterer = hier_clusterer.fit(data)

plt.title('Complete Hierarchical Clustering Dendrogram (No scaling)')
# plot the top three levels of the dendrogram
plot_dendrogram(hier_clusterer, truncate_mode='level')
plt.xlabel("State")
plt.show()
```

<img src="../../Images/Chapter10/problem9_parta_dendogram.png" alt="bootstrap_limit" title="bootstrap_limit"  />

### Part b)
Cut the dendogram off at three distinct clusters.

```python
hier_clusterer = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', linkage='complete', distance_threshold=None, n_clusters=3)
hier_clusterer = hier_clusterer.fit(data)
```

Return the clusters and reindex with the index dictionary

```python
cluster_labels = {i: np.where(heir_clusterer.labels_ == i)[0] for i in range(heir_clusterer.n_clusters)}

for c, state in cluster_labels.items():
    state_clusters = [index_dict[st] for st in state]
    print("Group ", c)
    print("--------")
    for st in state_clusters:
        print(st)
    print('\n')
```

```
Group  0
--------
Arkansas
Connecticut
Delaware
Hawaii
Idaho
Indiana
Iowa
Kansas
Kentucky
Maine
Massachusetts
Minnesota
Missouri
Montana
Nebraska
New Hampshire
New Jersey
North Dakota
Ohio
Oklahoma
Oregon
Pennsylvania
Rhode Island
South Dakota
Utah
Vermont
Virginia
Washington
West Virginia
Wisconsin
Wyoming


Group  1
--------
Alabama
Alaska
Georgia
Louisiana
Mississippi
North Carolina
South Carolina
Tennessee


Group  2
--------
Arizona
California
Colorado
Florida
Illinois
Maryland
Michigan
Nevada
New Mexico
New York
Texas
```

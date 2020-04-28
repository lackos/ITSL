# Chapter 8 Applied Problems

## Problem 7
Load in the dataframe and separete the target and predictors.
```python
import pandas as pd
import numpy as np
import os
boston_df = pd.read_csv(os.path.join(DATA_DIR, "boston.csv"))
X = boston_df.drop('target', axis=1)
y = boston_df['target']
```

Instantiate the random forest regressor

```python
from sklearn.ensemble import RandomForestRegressor
f_model = RandomForestRegressor()
```

#### Number of features considered
We will first find the ideal maximium number of predictors to consider at each split. In sklearn this is controller by the `max_features` variable. We iterate over this in a for loop, cross-validating the model and storing the training and testing errors in a list. The scores are standard $R^2$ values.

```python
## Create an array of integers to iterate over
num_feats = np.arange(1,X.shape[1])
## Set the number of trees to be constant.
f_model.set_params(n_estimators=50)
## Create empty lists for training and testing scores
test_cv_scores_feats = []
train_cv_scores_feats = []
## Loop over the number of features
for i in num_feats:
    print(i)
    ## Update the regressor and cross_validate
    f_model.set_params(max_features = i)
    scores = cross_validate(f_model, X, y, return_train_score=True)
    test_cv_scores_feats.append(scores['test_score'].mean())
    train_cv_scores_feats.append(scores['train_score'].mean())
```

The results are plotted in the figure below. We can see that the training score consistently performs very well, even when considering only 1 feature. However the training set is affected quite significantly with maximum performance of $R^2_{test} \approx 0.7$ for 7 features considered. In this model we have fixed the number of trees at $50$, there may be different outcomes for the models dependence if a different value was used.

#### Number of trees used in the forest
Next we will study how the number of decision trees used in aggregating the forest affects its performance.

```python
## Create an array of integers to iterate over
num_trees = np.arange(1,50,1)
## Set the number of predictors considered at each step to be constant. In this case we use the total number of predictors
f_model.set_params(max_features = X.shape[1])
## Create empty lists for training and testing scores
test_cv_scores_ntrees = []
train_cv_scores_ntrees = []
## Loop over the number of trees
for trees in num_trees:
    print(trees)
    f_model.set_params(n_estimators = trees)
    scores = cross_validate(f_model, X, y, return_train_score=True)
    test_cv_scores_ntrees.append(scores['test_score'].mean())
    train_cv_scores_ntrees.append(scores['train_score'].mean())
```

The results are plotted in the figure below.

Here again the training score performs extremely well for all tree numbers considered. The test score starts off poorly but quite reaches uniformity for N-trees=4 and does not significantly change after that. However, this test score is quite poor at $R^2 \approx 60$.


<img src="../Images/Chapter8/q7_model_performance.png" alt="PRSS Ridge" title="Boxplots of weekly_df"  />

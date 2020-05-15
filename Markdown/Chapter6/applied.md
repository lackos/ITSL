# Chapter Six Applied Problems
```python
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn sns

np.random.seed(100)
```

## Problem Eight
### Parts a) & b)
To generate the reponse predictor and response the following code was used:

``` python
## Generate the simulated data
X = np.random.randn(100)
err = np.random.randn(100)

## Set the estimators
beta_0 = 10
beta_1 = 20
beta_2 = 20
beta_3 = 3

## Generate the response
Y = beta_0 + beta_1*X + beta_2*X**2 + beta_3*X**3 + err
```

### Part c)
Python does not have an inbuilt best subset selection method. Therefore a simple function `bss` had to be created which, given a set of predictors, returns a model of the best subset for some metric. See the script for the function.

Using this function and a set of predictors of X up to the power of 10 we find,

```
Best subset of predictors for Adjusted R squared ('X_1', 'X_2', 'X_5', 'X_7', 'X_9')
Best subset of predictors for BIC ('X_1', 'X_2', 'X_5', 'X_7', 'X_9')
Best subset of predictors for AIC ('X_1', 'X_2', 'X_5', 'X_7', 'X_9')
```

Here we see that, for this random seed, the best subset was selected for each one. However, this is not ideal as we know the 'True' value is only to the power of three. The fitted model is too flexible, fitting the noise, and may not perform well on a test set.

### Part d)
Once again sklearn or statsmodels does not have forward stepwise selection or backwards stepwise seletion.

## Problem Nine
### Part a)
Split the data in test set and training set (this will only be used for the linear regression, all other models are cross-validated with the whole set).

```python
import pandas as pd
import numpy as np
## Load the dataset
college_df = pd.read_csv("college.csv")
college_df = college_df.set_index('Unnamed: 0')

## One-hot encode the categorical variable
college_df = pd.get_dummies(college_df, drop_first=True)

## Split into predictors and target
y = college_df['Apps']
X = college_df.drop('Apps', axis=1)

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```
#### Note on other feature engineering
For each numerical predictor $X$ we also include predictors $X^2, X^3$ and $X^4$. We will do a run of each model with the extra variables and the vanilla set.
The code for the further feature engineering is:

```python
## For each original numerical predictor X, create cols X^2, X^3 and X^4
for col in college_df.columns:
    if col == 'Private_Yes' or col == 'Apps':
        pass
    else:
        for i in [2,3,4]:
            new_col = col + '_' + str(i)
            college_df[new_col] = college_df[col].apply(lambda x: np.power(x,i))
```

### Part b)
Simple Linear model
``` python
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

## Ordinary Least Squares model
X_train = X_train.values
y_train = y_train.values

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
ols = sm.OLS(y_train, X_train).fit()
# print(ols.summary())
predictions = ols.predict(X_test)
error = mean_squared_error(y_test, predictions, squared=False)
print("least squares test error (RMSE): ", round(error,2))
```

With feature engineering

```
least squares test error (RMSE):  7148.37
```

Without feature engineering

```
least squares test error (RMSE):  1348.33
```

### Part c)
Ridge regression model.

```python
## Ridge Regression model

## Set baseline RMSE score
best_score = 10000

## Set empty cv_scores for plot
cv_scores_ridge = []

## Instantiate the ridge regressor
ridge = Ridge(tol=0.1, normalize=True)

## Loop over alpha values to find optimal cross-validated test error
for lamb in np.arange(1,2000,1):
    ridge.set_params(alpha=lamb)
    scores = cross_validate(ridge, X, y, scoring='neg_root_mean_squared_error')
    cv_scores_ridge.append(scores['test_score'].mean())

    ## Test if cv score beat previous best
    if np.abs(scores['test_score'].mean()) < best_score:
        ridge.fit(X,y)
        best_score = scores['test_score'].mean()
        best_lamb = lamb
        best_model_coef = ridge.coef_
print("Best Ridge Test score: ", round(np.abs(best_score),2))
# print("Ridge coefficients", best_model_coef)
```

With feature engineering

```
Best Ridge Test score:  1393.68
```

Without feature engineering

```
Best Ridge Test score:  1631.28
```

### Part d)
Lasso Model

```python
## Lasso Regression model

## Set baseline RMSE score
best_score = 10000

## Set empty cv_scores for plot
cv_scores_lasso = []

## Instantiate the ridge regressor
lasso = Lasso(tol=0.1, normalize=True)

## Loop over alpha values to find optimal cross-validated test error
for lamb in np.arange(10,2000,1):
    lasso.set_params(alpha=lamb)
    scores = cross_validate(lasso, X, y, scoring='neg_root_mean_squared_error')
    cv_scores_lasso.append(scores['test_score'].mean())

    ## Test if cv score beat previous best
    if np.abs(scores['test_score'].mean()) < best_score:
        lasso.fit(X,y)
        best_score = scores['test_score'].mean()
        best_lamb = lamb
        best_model_coef = lasso.coef_

unique, counts = np.unique(best_model_coef, return_counts=True)
count_dict = dict(zip(unique, counts))
print("Best Lasso Test score: ", round(np.abs(best_score),2))
print(count_dict)
```

With feature engineering:

```
Best Lasso Test score:  1107.57
{0.0: 56, 3.668900482912154e-15: 1, 5.852424881999367e-11: 1, 1.30198199698816e-06: 1, 8.198389236748027e-06: 1, 0.0012739917135652873: 1, 0.01045029166083545: 1, 0.14155511551886565: 1, 1.3127127764098352: 1, 8.886578403589283: 1}
```
There are 8 non-zero coefficients

Without Feature Engineering:

```
Best Lasso Test score:  1194.42
{0.0: 14, 0.014199601111647397: 1, 1.3431347761848054: 1, 20.671942456008864: 1}
```
There are only 3 nonzero components

### Part e)
Principal component regression with ordinary least squares

```python
## Principal Component Regression Model
best_score = 10000

## Scale predictors and break down data set into principal components
scaler = StandardScaler()
pca = PCA()

X_scaled = scaler.fit_transform(X)
X_PCA = pca.fit_transform(X_scaled)

## Set empty score list
cv_scores_PCR = []

## Instantiate the ordinary linear regressor
ols = LinearRegression()

## Iterate over number of principal components used
for i in range(1,X_PCA.shape[1]):
    ## Select the first i principal components to train
    X_train = X_PCA[:,0:i]
    scores = cross_validate(ols, X_train, y, scoring='neg_root_mean_squared_error')
    cv_scores_PCR.append(scores['test_score'].mean())
    ## Test if cv score beat previous best
    if np.abs(scores['test_score'].mean()) < best_score:
        best_score = np.abs(scores['test_score'].mean())
        best_num_PC = i
print("Best PCR Test score: ", round(np.abs(best_score),2))
print("Number of Principal Components", best_num_PC)
```

With Feature engineering:

```
Best PCR Test score:  1164.84
Number of Principal Components 35
```

Without feature engineering:

```
Best PCR Test score:  1209.48
Number of Principal Components 16
```

### Part f)
Partial Least Squares

```python
## Partial Least Squares Approach

## Set baseline RMSE score
best_score = 10000

## Instantiate preprocessing
pls = PLSRegression()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Set empty score list
cv_scores_PLS = []
for i in range(2,64):
    pls.set_params(n_components=i)
    scores = cross_validate(pls, X_scaled, y, scoring='neg_root_mean_squared_error')
    if np.abs(scores['test_score'].mean()) < best_score:
        best_score = np.abs(scores['test_score'].mean())
        best_num_PC = i
print("Best PLS Test score: ", round(np.abs(best_score),2))
print("Number of Latent variables", best_num_PC)
```

With Feature engineering:

```
Best PLS Test score:  1146.62
Number of Latent variables 7
```

Without Feature engineering

```
Best PLS Test score:  1183.92
Number of Latent variables 14
```

### Part g)
The complete results are in the following table. All tests are RMSEs for ease of camparison.

Model | With FE | Without FE
:-----|:----------:|:------:
OLS  | 7148  | 1348
Ridge  | 1393  | 1631
Lasso  | 1108  | 1194
PCR  | 1165  |  1209
PLS  | 1147  |  1184

From this table we can see that both Lasso and PLS are the best performing models. In both of these models the inclusion of the higher order predictors has a negligible increase in performance. It should be noted that we did not treat the training set completely correct with the PLS and PCR models. In these models we fitted the scaler and the PCA to the entire training set before cross-validation. This will have resulted in so test-train contamination. Ideally, these models would be rewritten as pipelines and the scaler/PCA refitted to the $k-1$ cv folds and the testing cv fold transformed accordingly each time.This may increase the reported test errors for PCR and PLS.


It is interesting to note how poorly the OLS performed with the extra predictors. This is because, without cross validation, all the unregularized predictors were considered which resulted in an extremely flexible model which overfit  the training set and gave poor test set results. With the extra features the OLS still performed relatively poorly, but significantly better that with the extra features.

There is not too much difference between the test errors for Ridge and Lasso and the both perform relatively well with lasso consistently better.

It is also interesting to compare the performance of PCR and PLS as they are based on similar concepts with PLS being more sophisticated as the principal component selection is supervised. Despite this, for this data set, there is no significant different in performance between the two models.

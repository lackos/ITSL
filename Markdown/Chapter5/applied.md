# Chapter 5: Resampling Methods
# Applied Problems

```python
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn sns

np.random.seed(100)
```
## Problem Five
### Part a)
Load the data
```python
## Load the default dataset
default_df = pd.read_csv(os.path.join(DATA_DIR, 'default.csv'))
```
Fit the logisitic regression model.
```python
from sklearn.linear_model import LogisticRegression
## Fit the logistic regression (no train test split or cross val)
X = default_df[['income', 'balance']]
y = default_df['default']
log_reg_model = LogisticRegression()
log_reg_model.fit(X, y)
```
### Part b)
Normally we would use sklearn's `train_test_split` to generate the training and test set, however here we do it manually for demonstration. Throughout the rest of the problems we will use sklearn.
#### i.
```python
### Shuffle the data
index = default_df.index
default_df = shuffle(default_df)
default_df.index = index
## Set train set to 80% and validation to 20%
size = default_df.shape[0]
train = default_df.loc[0:0.8*size - 1]
validation = default_df.loc[0.8*size:]
```
#### ii.
Fit the model to the training set:
```python
## Fit logistic regression to the training set
X_train = train[['income', 'balance']]
X_val = validation[['income', 'balance']]
y_train = train['default']
y_val = validation['default']
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
```
#### iii.
Sklearn's default classification threshold is 50% and therefore a simple predict method is all that is needed. To see how to do this manually refer the solution of CHapter 4 problems.
```python
preds = log_reg_model.predict(X_val)
```
#### iv.
Compute the test error with the trace of the confusion matrix and the total number of predictions.
```python
## iv) Compute the test error
cm = confusion_matrix(preds, y_val)
test_error = (cm[0,1] + cm[1,0])/(len(preds))
print('Test Error: ',test_error)
```
```
Test Error: 0.036
```
### Part c)
This can be accomplished with three different random seeds to get the following results,
```
# Random Seed 101
Test Error:  0.0265
# Random Seed 102
Test Error:  0.033
# Random Seed 103
Test Error:  0.0245
```
Here we see that the results are different by similar for the 4 different splits. On average, this model has a test error of 0.03.

### Part d)
Refit the model with the extra variable and generate the new predictions.
```python
## Fit logistic regression to the training set
X_train = train[['income', 'balance', 'student_Yes']]
X_val = validation[['income', 'balance', 'student_Yes']]
y_train = train['default']
y_val = validation['default']
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
preds = log_reg_model.predict(X_val)

## Compute the test error
cm = confusion_matrix(preds, y_val)
test_error = (cm[0,1] + cm[1,0])/(len(preds))
print('Test Error: ',test_error)
```
```
Test Error:  0.036
```

## Problem Six
### Part a)
```python
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Load the default dataset
default_df = pd.read_csv(os.path.join(DATA_DIR, 'default.csv'))
binary_dict = {'No':0, 'Yes':1}
default_df['default_bin'] = default_df['default'].map(binary_dict)

X = default_df[['income', 'balance']]
y = default_df['default_bin']

##Instantiate and fit the logistic regression with a single variable
logit_reg = smf.logit(formula = "default_bin ~ income + balance", data= default_df).fit()
print(logit_reg.summary())
```

```
                        Logit Regression Results
==============================================================================
Dep. Variable:            default_bin   No. Observations:                10000
Model:                          Logit   Df Residuals:                     9997
Method:                           MLE   Df Model:                            2
Date:                Thu, 02 Apr 2020   Pseudo R-squ.:                  0.4594
Time:                        10:27:20   Log-Likelihood:                -789.48
converged:                       True   LL-Null:                       -1460.3
Covariance Type:            nonrobust   LLR p-value:                4.541e-292
==============================================================================
coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -11.5405      0.435    -26.544      0.000     -12.393     -10.688
income      2.081e-05   4.99e-06      4.174      0.000     1.1e-05    3.06e-05
balance        0.0056      0.000     24.835      0.000       0.005       0.006
==============================================================================
```



## Problem Eight
### Parts c) - d)
Random seed = 30
#### Standard Error tables

|Intercept |$X$         |$X^2$       |$X^3$       |$X^4$       |
|----------|----------|----------|----------|----------|
|0.263     |0.246     |          |          |          |
|0.127     |0.096     |0.074     |          |          |
|0.14      |0.16      |0.099     |0.0555    |          |
|0.152     |0.21      |0.179     |0.0949    |0.044     |

Random Seed =100

|Intercept |$X$         |$X^2$       |$X^3$       |$X^4$       |
|----------|----------|----------|----------|----------|
|0.235     |0.241     |          |          |          |
|0.139     |0.111     |0.096     |          |          |
|0.141     |0.201     |0.1       |0.0765    |          |
|0.163     |0.219     |0.241     |0.0881    |0.06      |

While the two random seeds generate std. errors of the same order of magnitude, they are still consistently different.

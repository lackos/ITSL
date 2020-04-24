# Chapter 5 Applied Problems
## Problem 6:
### a)
```python
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

### b)

## Problem 8
### c) - d)
Random seed = 30
#### Standard Error tables
|Intercept |X         |X^2       |X^3       |X^4       |
|----------|----------|----------|----------|----------|
|0.263     |0.246     |          |          |          |
|0.127     |0.096     |0.074     |          |          |
|0.14      |0.16      |0.099     |0.0555    |          |
|0.152     |0.21      |0.179     |0.0949    |0.044     |

Random Seed =100
|Intercept |X         |X^2       |X^3       |X^4       |
|----------|----------|----------|----------|----------|
|0.235     |0.241     |          |          |          |
|0.139     |0.111     |0.096     |          |          |
|0.141     |0.201     |0.1       |0.0765    |          |
|0.163     |0.219     |0.241     |0.0881    |0.06      |

While the two random seeds generate std. errors of the same order of magnitude, they are still consistently different.

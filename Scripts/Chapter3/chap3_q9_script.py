import pandas as pd
import numpy as np

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter3')

def residual_plot(smf_model, num_max_res=3, ax=None):
    ###
    ### https://towardsdatascience.com/going-from-r-to-python-linear-regression-diagnostic-plots-144d1c4aa5a
    ###
    if ax is None:
        ax = plt.gca()
    residuals = smf_model.resid
    fitted = smf_model.fittedvalues
    smoothed = lowess(residuals,fitted)
    top3 = abs(residuals).sort_values(ascending = False)[:num_max_res]

    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8,7)
    ax.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none')
    ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Fitted Values')
    ax.set_title('Residuals vs. Fitted')
    ax.plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3)

    for i in top3.index:
        ax.annotate(i,xy=(fitted[i],residuals[i]))

    # plt.show()
    return ax

def qq_plot(smf_model, num_max_res=3, ax=None):
    if ax is None:
        ax = plt.gca()
    sorted_student_residuals = pd.Series(smf_model.get_influence().resid_studentized_internal)
    sorted_student_residuals.index = smf_model.resid.index
    sorted_student_residuals = sorted_student_residuals.sort_values(ascending = True)
    df = pd.DataFrame(sorted_student_residuals)
    df.columns = ['sorted_student_residuals']
    df['theoretical_quantiles'] = stats.probplot(df['sorted_student_residuals'], dist = 'norm', fit = False)[0]
    rankings = abs(df['sorted_student_residuals']).sort_values(ascending = False)
    top3 = rankings[:num_max_res]

    x = df['theoretical_quantiles']
    y = df['sorted_student_residuals']
    ax.scatter(x,y, edgecolor = 'k',facecolor = 'none')
    ax.set_title('Normal Q-Q')
    ax.set_ylabel('Standardized Residuals')
    ax.set_xlabel('Theoretical Quantiles')
    ax.plot([np.min([x,y]),np.max([x,y])],[np.min([x,y]),np.max([x,y])], color = 'r', ls = '--')
    for val in top3.index:
        ax.annotate(val,xy=(df['theoretical_quantiles'].loc[val],df['sorted_student_residuals'].loc[val]))
    # plt.show()
    return ax

def scale_location_plot(smf_model, num_max_res=3, ax=None):
    if ax is None:
        ax = plt.gca()
    student_residuals = smf_model.get_influence().resid_studentized_internal
    sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
    sqrt_student_residuals.index = smf_model.resid.index
    fitted = smf_model.fittedvalues
    smoothed = lowess(sqrt_student_residuals,fitted)
    top3 = abs(sqrt_student_residuals).sort_values(ascending = False)[:num_max_res]

    ax.scatter(fitted, sqrt_student_residuals, edgecolors = 'k', facecolors = 'none')
    ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
    ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
    ax.set_xlabel('Fitted Values')
    ax.set_title('Scale-Location')
    ax.set_ylim(0,max(sqrt_student_residuals)+0.1)
    for i in top3.index:
        ax.annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
    # plt.show()
    return ax

def leverage_plot(smf_model, num_max_res=3, ax=None):
    if ax is None:
        ax = plt.gca()
    student_residuals = pd.Series(smf_model.get_influence().resid_studentized_internal)
    student_residuals.index = smf_model.resid.index
    df = pd.DataFrame(student_residuals)
    df.columns = ['student_residuals']
    df['leverage'] = smf_model.get_influence().hat_matrix_diag
    smoothed = lowess(df['student_residuals'],df['leverage'])
    sorted_student_residuals = abs(df['student_residuals']).sort_values(ascending = False)
    top3 = sorted_student_residuals[:num_max_res]

    x = df['leverage']
    y = df['student_residuals']
    xpos = max(x)+max(x)*0.01
    ax.scatter(x, y, edgecolors = 'k', facecolors = 'none')
    ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
    ax.set_ylabel('Studentized Residuals')
    ax.set_xlabel('Leverage')
    ax.set_title('Residuals vs. Leverage')
    ax.set_ylim(min(y)-min(y)*0.15,max(y)+max(y)*0.15)
    ax.set_xlim(-0.01,max(x)+max(x)*0.05)
    plt.tight_layout()
    for val in top3.index:
        ax.annotate(val,xy=(x.loc[val],y.loc[val]))

    cooksx = np.linspace(min(x), xpos, 50)
    p = len(smf_model.params)
    poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
    poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
    negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
    negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)

    ax.plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
    ax.plot(cooksx,poscooks05y, ls = ':', color = 'r')
    ax.plot(cooksx,negcooks1y, ls = ':', color = 'r')
    ax.plot(cooksx,negcooks05y, ls = ':', color = 'r')
    ax.plot([0,0],ax.get_ylim(), ls=":", alpha = .3, color = 'k')
    ax.plot(ax.get_xlim(), [0,0], ls=":", alpha = .3, color = 'k')
    ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
    ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
    ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
    ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
    ax.legend()
    return ax
    # plt.show()

def diagnostic_plot(smf_model, num_max_res=3, filename=None):
    plt.style.use('seaborn') # pretty matplotlib plots
    plt.rc('font', size=10)
    plt.rc('figure', titlesize=13)
    plt.rc('axes', labelsize=10)
    plt.rc('axes', titlesize=13)
    plt.rc('legend', fontsize=8)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    residual_plot(smf_model, num_max_res, ax = ax1)
    qq_plot(smf_model, num_max_res, ax = ax2)
    scale_location_plot(smf_model, num_max_res, ax = ax3)
    leverage_plot(smf_model, num_max_res, ax = ax4)
    if filename == None:
        plt.show()
    else:
        plt.savefig(os.path.join(IMAGE_DIR,filename), format='png', dpi=500)

def part_a():
    ## Import auto.csv from DATA_DIR
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))


    ### a) Scatter plot matrix of auto dataset
    ## 'horsepower' contains some non-numeric rows which need to be removed
    ## First set all non-numeric rows to 'nan' then remove them
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ## Plot the scatter matrix
    ax = sns.pairplot(auto_df)
    plt.gcf().subplots_adjust(bottom=0.05, left=0.1, top=0.95, right=0.95)
    ax.fig.suptitle('Auto Quantitative Scatter Matrix', fontsize=35)
    plt.savefig(os.path.join(IMAGE_DIR,'auto_scatter_matrix.png'), format='png', dpi=250)
    plt.show()
    plt.close()

def part_b():
    ## Import auto.csv from DATA_DIR
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))

    ## 'horsepower' contains some non-numeric rows which need to be removed
    ## First set all non-numeric rows to 'nan' then remove them
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ### b) Correlation matrix of auto dataset.
    corr = auto_df.corr()
    sns.set(font_scale=1.25)
    hm = sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
    plt.show()
    plt.savefig(os.path.join(IMAGE_DIR,'q9_auto_corr_matrix.png'), format='png', dpi=500)

def part_c():
    ## Import auto.csv from DATA_DIR
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))


    ### a) Scatter plot matrix of auto dataset
    ## 'horsepower' contains some non-numeric rows which need to be removed
    ## First set all non-numeric rows to 'nan' then remove them
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ## Drop the 'name' column
    auto_df.drop('name', axis=1, inplace=True)

    ## Set the target and predictors
    X = auto_df.drop('mpg', axis=1)
    y = auto_df['mpg']

    ## Reshape the columns in the required dimensions for sklearn
    length = X.values.shape[0]
    y = y.values.reshape(length, 1)

    lm = LinearRegression()
    lm.fit(X,y)
    print(lm.coef_)

    ## Doe the same with statsmodels
    feature_string = ' + '.join(X.columns)
    results = smf.ols("mpg ~ " + feature_string, data=auto_df).fit()
    print(results.summary())

def part_d():
    ## Import auto.csv from DATA_DIR
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))


    ### a) Scatter plot matrix of auto dataset
    ## 'horsepower' contains some non-numeric rows which need to be removed
    ## First set all non-numeric rows to 'nan' then remove them
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ### c) Perform a multiple linear regression with mpg as the target and all other variable as prodictors
    ## Drop the 'name' column
    auto_df.drop('name', axis=1, inplace=True)

    ## Set the target and predictors
    X = auto_df.drop('mpg', axis=1)
    y = auto_df['mpg']

    feature_string = ' + '.join(X.columns)
    results = smf.ols("mpg ~ " + feature_string, data=auto_df).fit()
    diagnostic_plot(results, num_max_res=3)

def part_e():
    ## Import auto.csv from DATA_DIR
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))


    ### a) Scatter plot matrix of auto dataset
    ## 'horsepower' contains some non-numeric rows which need to be removed
    ## First set all non-numeric rows to 'nan' then remove them
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ### c) Perform a multiple linear regression with mpg as the target and all other variable as prodictors
    ## Drop the 'name' column
    auto_df.drop('name', axis=1, inplace=True)

    ## Set the target and predictors
    X = auto_df.drop('mpg', axis=1)
    y = auto_df['mpg']

    ## Create interaction effects
    feature_names = X.columns
    poly = PolynomialFeatures(interaction_only=True,include_bias = False)
    X = poly.fit_transform(X)

    ## Create new linear model with new features
    ## Recreate dataframe out of preporcessed numpy array
    column_names = poly.get_feature_names(input_features=feature_names)
    columns = [name.replace(' ', '_') for name in column_names]
    X = pd.DataFrame(data=X, columns=columns)
    full_df = X.join(auto_df['mpg'])
    feature_string = ' + '.join(columns)
    formula = "mpg ~ " + feature_string
    results = smf.ols("mpg ~ " + feature_string, data=full_df).fit()
    print(results.summary())

def main():
    ###
    ### Question 9
    ###

    ## Import auto.csv from DATA_DIR
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))


    ### a) Scatter plot matrix of auto dataset
    ## 'horsepower' contains some non-numeric rows which need to be removed
    ## First set all non-numeric rows to 'nan' then remove them
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ### c) Perform a multiple linear regression with mpg as the target and all other variable as prodictors
    ## Drop the 'name' column
    auto_df.drop('name', axis=1, inplace=True)

    ## Set the target and predictors
    X = auto_df.drop('mpg', axis=1)
    y = auto_df['mpg']

    ## Reshape the columns in the required dimensions for sklearn
    length = X.values.shape[0]
    y = y.values.reshape(length, 1)

    lm = LinearRegression()
    lm.fit(X,y)
    print(lm.coef_)

    ## Doe the same with statsmodels
    feature_string = ' + '.join(X.columns)
    results = smf.ols("mpg ~ " + feature_string, data=auto_df).fit()
    print(results.summary())

    ## e) Create interaction effects
    feature_names = X.columns
    poly = PolynomialFeatures(interaction_only=True,include_bias = False)
    X = poly.fit_transform(X)
    # print(X.shape)
    # print(poly.get_feature_names(input_features=feature_names))

    ## Create new linear model with new features
    ## Recreate dataframe out of preporcessed numpy array
    column_names = poly.get_feature_names(input_features=feature_names)
    columns = [name.replace(' ', '_') for name in column_names]
    X = pd.DataFrame(data=X, columns=columns)
    full_df = X.join(auto_df['mpg'])
    # print(pd.concat([auto_df['mpg'], X], axis=1))
    # print(X)
    feature_string = ' + '.join(columns)
    # print(feature_string)
    formula = "mpg ~ " + feature_string
    print(formula)
    results = smf.ols("mpg ~ " + feature_string, data=full_df).fit()
    print(results.summary())




if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
IMAGE_DIR = os.path.join(os.path.join(BASE_DIR, 'Images'), 'Chapter3')

###
### Question 8
###

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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
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

    ### a) fit a linear regression of horsepower to mpg.
    ## 'horsepower' contains some non-numeric rows which need to be removed
    ## First set all non-numeric rows to 'nan' then remove them
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ## Set the target and predictors
    X = auto_df['horsepower']
    y = auto_df['mpg']

    ## Reshape the columns in the required dimensions for sklearn
    length = X.values.shape[0]
    X = X.values.reshape(length, 1)
    y = y.values.reshape(length, 1)


    ## Initiate the linear regressor and fit it to data using sklearn
    regr = LinearRegression()
    regr.fit(X, y)
    pred_y = regr.predict(X)

    ## Output statistical summary using statsmodels
    results = smf.ols('mpg ~ horsepower', data=auto_df).fit()
    print(results.summary())

def part_b():
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ## Set the target and predictors
    X = auto_df['horsepower']
    y = auto_df['mpg']

    ## Reshape the columns in the required dimensions for sklearn
    length = X.values.shape[0]
    X = X.values.reshape(length, 1)
    y = y.values.reshape(length, 1)


    ## Initiate the linear regressor and fit it to data using sklearn
    regr = LinearRegression()
    regr.fit(X, y)
    pred_y = regr.predict(X)

    ### Plot the linear model
    ## Instantiate the figure and axes
    fig, ax = plt.subplots(figsize=(15,15))

    ## Scatter plotwith fitted line
    plt.scatter(X, y, color='grey')
    line, = ax.plot(X, pred_y, color='red', linewidth=2)
    plt.rc('legend', fontsize=8)

    plt.title("Linear model of mpg vs horsepower", fontsize=30)
    ax.set_xlabel('Horsepower', fontsize=25)
    ax.set_ylabel('mpg', fontsize=25)
    line.set_label("y = " + str(round(float(regr.intercept_), 2)) + " " + str(round(float(regr.coef_), 2)) + "x", )
    ax.legend(fontsize=25)

    fig.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR,'q8_mpg_hp_lr.png'), format='png', dpi=500)

    # plt.show()
    plt.close()

def part_c():
    ## Import auto.csv from DATA_DIR
    auto_df = pd.read_csv(os.path.join(DATA_DIR, 'auto.csv'))

    ### a) fit a linear regression of horsepower to mpg.
    ## 'horsepower' contains some non-numeric rows which need to be removed
    ## First set all non-numeric rows to 'nan' then remove them
    auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
    auto_df['mpg'] = pd.to_numeric(auto_df['mpg'], errors='coerce')
    auto_df.dropna(subset= ['horsepower', 'mpg',], inplace=True)

    ## Set the target and predictors
    X = auto_df['horsepower']
    y = auto_df['mpg']

    ## Reshape the columns in the required dimensions for sklearn
    length = X.values.shape[0]
    X = X.values.reshape(length, 1)
    y = y.values.reshape(length, 1)


    ## Initiate the linear regressor and fit it to data using sklearn
    regr = LinearRegression()
    regr.fit(X, y)
    pred_y = regr.predict(X)

    ## Output statistical summary using statsmodels
    results = smf.ols('mpg ~ horsepower', data=auto_df).fit()
    print(results.summary())


    # Plot residuals
    fig, ax = plt.subplots()
    sns.residplot(X, y, lowess=True, color="g", ax=ax)

    plt.title("Linear model of mpg vs horsepower", fontsize=30)
    ax.set_xlabel('Fitted values (mpg vs horsepower)', fontsize=20)
    ax.set_ylabel('residuals', fontsize=20)

    height = 3.403
    width = 1.518*height
    fig.set_size_inches(width, height)
    fig.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR,'mpg_hp_residuals.png'), format='png', dpi=500)
    plt.show()
    plt.close()

    diagnostic_plot(results, num_max_res=3, filename='q8_mpg_horsepower_diagplot.png')

def main():
    part_a():
    part_b():
    part_c():




if __name__ == "__main__":
    main()

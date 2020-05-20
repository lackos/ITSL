# Chapter Three: Linear Regression
# Conceptual Problems

## Problem One
The null hypothesis of this model can be stated as:
"In determining the affect advertising options has on sales, there is no effect
caused by TV, radio or newspaper advertising."

Due to the statistically significant p-values of TV and radio advertising we can
 reject the null hypothesis and assume that they have a statistically significant
 affect on sales. We can not make a similar conclusion for newspaper advertising
 as it is not statistically significant. This does not make any conclusion on the
 interaction effect newspapers may have on other predictors.

## Problem Two
The KNN classifier will classify the test point $x_{0}$ based on the probability
 calculated from the $k$ nearest points. KNN regression on the other hand will
 assign the test point $x_0$ the average value of the $k$ nearest neighbours.

## Problem Three
### Part a)
The regression equation can be split into two separate regression lines for
 males and females as (all hats on estimators are omitted):

 \[
 Y_{male} = \beta_{0} + \beta_{1}X_{1}  + \beta_{2}X_{2}  + \beta_{4}X_{1}X_{2}
 \]

 \[
 Y_{female} = \beta_{0} + \beta_{1}X_{1}  + \beta_{2}X_{2}  + \beta_{3} +  \beta_{4}X_{1}X_{2} + \beta_{5}X_{1}X_{3}
 \]

 Therefore the response of females can be written in terms of the response of males as,
 \[
 Y_{female} = Y_{male}  + \beta_{3} +  \beta_{5}X_{1}X_{3}
 \]

 From this equation we see that females making more that males is dependent on
 the interaction between the GPA and IQ of the person. For a woman to make more
 than a man
 \[
 X_{1}>-\frac{\beta_{3}}{\beta_{5}X_{3}}
 \]

### Part b)
$Y_{female} = \$137,100$

### Part c)
False. The significance can only be determined with a statistic. The magnitude
 of an estimator offers no insight on its validity in the model. For example, we
 could change the order of magnitude of an estimator be any degree with a change
 of units.

 ## Problem Four
This entire problem can be understood in the context of bias-variance tradeoff.
 ### Part a)
 *THe training RSS would be lower for the cubic model.* As the cubic model is more flexible than the linear model it will fit the training set better than the linear model.

 ### Part b)
 *The test RSS will be lower for the linear model.* As the true relationship in linear the the flexible model will overfit the training data (higher variance) and perform poorly on the test set. Conversely, the linear model will have both low bias and variance and have a lower test RSS.

 ### Part c)
*THe training RSS would be lower for the cubic model.* This is for the same reasons as part a).

### Part d)
 *Cannot definitively say*. The high bias of the  linear model will result in a poor RSS if the relation is very non-linear and the cubic model will have a better fit. However, if the true relationship is close to linear the linear model may perform better due to the high variance of the cubic model.

 ## Problem Five

Linear model with no intercept,
\[
y_{i} = \beta x_{i} \rightarrow y_i = \sum_{j=1}^{n} a_{j,i} y_{j}
\]
Where,
\[
\beta = \dfrac{\sum_{j=1}^{n}x_jy_j}{\sum_{k=1}^{n}x_{k}^2}.
\]

Substituting into the equation,

\[
\begin{aligned}
y_{i} &= \dfrac{\sum_{j=1}^{n}x_jy_j}{\sum_{k=1}^{n}x_{k}^2} x_{i} \\
&= \sum_{j=1}^{n} \dfrac{x_{i}x_{j}}{\sum_{k=1}^{n}x^2_k}y_j
\end{aligned}
\]
Therefore,
\[
\begin{aligned}
y_{i} &= \sum_{j=1}^{n} a_{j,i} y_{j}, \\
\text{Where,} \\
a_{j,i} &= \dfrac{x_{i}x_{j}}{\sum_{k=1}^{n}x^2_k}
\end{aligned}
\]

## Problem Six
*Show that a simple least squares line will pass through the average point* ($\bar{x}$, $\bar{y}$).

The average points are defined as,

\[
\bar{x} = \dfrac{\sum_{i=1}^{n}x_{i}}{n}, \ \bar{y} = \dfrac{\sum_{i=1}^{n}y_{i}}{n}
\]

THe least square model for the data  is given by,
\[
y_{i} = \beta_{0} + \beta_{1}x_{i}
\]
where,

\[
\beta_{0} = \bar{y} - \beta_{1}\bar{x} \\
\beta_{1} = \sum_{i=1}^{n} \dfrac{\left(x_{i} - \bar{x}\right)\left(y_{i} - \bar{y}\right)}{\left(x_{i} - \bar{x}\right)^2}
\]

We need to show that $\bar{y} = \beta_{0} + \beta_{1}\bar{x}$ or $\bar{y} - \beta_{0} - \beta_{1}\bar{x} = 0$.

Substituting in $\beta_0$

\[
\begin{aligned}
\bar{y} - \beta_{0} - \beta_{1}\bar{x} & = \bar{y} - \bar{y} - \beta_{1}\bar{x} - \beta_{1}\bar{x} \\
&= 0.
\end{aligned}
\]

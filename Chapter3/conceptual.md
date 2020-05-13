# Chapter 3 conceptual questions

1. The null hypothesis of this model can be stated as:
"In determining the affect advertising options has on sales, there is no effect
caused by TV, radio or newspaper advertising."

Due to the statistically significant p-values of TV and radio advertising we can
 reject the null hypothesis and assume that they have a statistically significant
 affect on sales. We can not make a similar conclusion for newspaper advertising
 as it is not statistically significant. This does not make any conclusion on the
 interaction effect newspapers may have on other predictors.

2. The KNN classifier will classify the test point $x_{0}$ based on the probability
 calculated from the $k$ nearest points. KNN regression on the other hand will
 assign the test point $x_0$ the average value of the $k$ nearest neighbours.

3a. The regression equation can be split into two separate regression lines for
 males and females as (all hats on estimators are omitted):

 $Y_{male} = \Beta_{0} + \Beta_{1}X_{1}  + \Beta_{2}X_{2}  + \Beta_{4}X_{1}X_{2}$

 $Y_{female} = \Beta_{0} + \Beta_{1}X_{1}  + \Beta_{2}X_{2}  + \Beta_{3} +  \Beta_{4}X_{1}X_{2} + \Beta_{5}X_{1}X_{3}$

 Therefore the response of females can be written in terms of the response of males as,
 $Y_{female} = Y_{male}  + \Beta_{3} +  \Beta_{5}X_{1}X_{3}$

 From this equation we see that females making more that males is dependent on
 the interaction between the GPA and IQ of the person. For a woman to make more
 than a man $X_{1}>-\frac{\Beta_{3}}{\Beta_{5}X_{3}}$

3b. $Y_{female} = \$137,100$

3c. False. The significance can only be determined with a statistic. The magnitude
 of an estimator offers no insight on its validity in the model. For example, we
 could change the order of magnitude of an estimator be any degree with a change
 of units.

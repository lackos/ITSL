## Conceptual Problems of Chapter 2

1a. **Flexible Model**. Due to the large number of datapoints that the training set accuratelyrepresents the predictors domain and therefore  overfitting is less of a concern and a flexible model midll be appropriate.

1b. **Non-Flexible**. For opposite reasons to 1a. A low number of samples means that the training data may be a good representation of the of the feature domain and therefore there is a high chance of overfitting. This is enhanced due to the large predictor space making the domain even less representative of possible values in the test set.

1c. **Flexible**. A highly non-linear relationship would mean that non-flexible models would have a large bias and therefore flexible models will be needed to capture the non-linearity.

1d. **Unknown**. In this case there is not much we can do as our models can only affect the reducible error or the model (In the model variance and bias) but, by definition, cannot affect the irreducible error. The most likely cause of this large variance in the error is missing a fundamental predictor in the data/model.

2.
 |	| Type of problem | n | p | inference | predictive |
   |----|-----------------|---|---|-----------|------------|
   |a)  | Regression      |500| 3 |     Y     |     N      |
   |b)  | Classification  |20 | 13|     N     |     Y      |
   |c)  | Regression      |52 | 4 |     N     |     Y      |		  
3. a) The variance, bias, test error, trianing error and Bayes error are plotted below.
<img src='../Images/Chapter2/flexibility_plots.png' width='150'> 
   b) **Variance**
      **Bias**
      **Training Error**
      **Test Error** 
      **Bayes Error**
	Fundamental error of the model also known as irreducible error.  
4. a) **Classification Problem**
   b) **Regression Problem**
   c) **Cluster Analysis**
5. | Flexible                       |       Non-Flexible         |
   |------------------------------ -|----------------------------|
   | Poor for inference             | Good for inference         |
   | High variance                  | Low variance               |
   | Low Bias                       | High Bias                  |
   | High chance of overfitting     | High chance of underfitting|
   | Difficult to interpret results | Easier to interpret results|

6. Parametric approaches are those which model a system with a set of statistical parameters. For example linear regression is a parametric model where the graadients and intercept are the parameters. A nonparametric approach is one which does not use these distinct parameters, for example a decsion tree model. Each has their advantages, parametric models are typically more interpretable but less flexible than non-parametric models and therefore have a higher bias. These non parametric forms do not assume a functional form of the relationship between the target and the predictors unline the parametric models. 
7a. Point | distance
   ------|---------
   1     | 3
   2     | 2
   3     | $\sqrt{10}$
   4     | $\sqrt{5}$
   5     | $\sqrt{2}$
   6     | $\sqrt{3}$
7b. For $K=1$ the nearest point is 5 and therefore it would be classified as 'Green'.
    For $K=3$ the nearest points are {'Red', 'Red', 'Green'} and therefore it will be classified as 'Red'.


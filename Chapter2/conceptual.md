## Conceptual Problems of Chapter 2

1a. **Flexible Model**. Due to the large number of datapoints that the training set accuratelyrepresents the predictors domain and therefore  overfitting is less of a concern and a flexible model midll be appropriate.

1b. **Non-Flexible**. For opposite reasons to 1a. A low number of samples means that the training data may be a good representation of the of the feature domain and therefore there is a high chance of overfitting. This is enhanced due to the large predictor space making the domain even less representative of possible values in the test set.

1c. **Flexible**. A highly non-linear relationship would mean that non-flexible models would have a large bias and therefore flexible models will be needed to capture the non-linearity.

d. **Unknown**. In this case there is not much we can do as our models can only affect the reducible error or the model (In the model variance and bias) but, by definition, cannot affect the irreducible error. The most likely cause of this large variance in the error is missing a fundamental predictor in the data/model.

2. |	| Type of problem | n | p | inference | predictive |
   |----|-----------------|---|---|-----------|------------|
   |a)  | Regression      |500| 3 |     Y     |     N      |
   |b)  | Classification  |20 | 13|     N     |     Y      |
   |c)  | Regression      |52 | 4 |     N     |     Y      |		  

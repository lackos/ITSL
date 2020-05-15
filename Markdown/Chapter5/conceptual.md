# Chapter Five Conceptual Problems

## Problem One
Given a combination of variable $X$ with variance '$\sigma_{X}$ and $Y$ with variance '$\sigma_{Y}$. For a combination of these variables $\alpha X + (1-\alpha) Y$ find value of $\alpha$ which minimizes the total variance $Var$.
\[
\begin{aligned}
Var &= Var(\alpha X + (1-\alpha) Y) \\
&= Var(\alpha X) + Var((1-\alpha) Y) + 2\times Cov(\alpha X, (1-\alpha) Y) \\
&= \alpha^2Var(X) + (1-\alpha)^2Var(Y) + 2\alpha(1-\alpha)Cov(X,  Y) \\
&= \alpha^2\sigma_X^2 + (1-\alpha)^2\sigma_Y^2 + 2\alpha\sigma_{XY} - 2\alpha^2\sigma_{XY}
\end{aligned}
\]
Minimizing this with respect to $\alpha$,
\[
\begin{aligned}
\dfrac{dVar}{d\alpha} &=  2\alpha\sigma_X^2 - 2(1-\alpha)\sigma_Y^2 + 2\sigma_{XY} - 4\alpha\sigma_{XY} \\
\\
\text{Setting this equal to 0} \\
\\
0 &= 2\alpha_{\text{min}}\sigma_X^2 - 2(1-\alpha_{\text{min}})\sigma_Y^2 + 2\sigma_{XY} - 4\alpha_{\text{min}}\sigma_{XY} \\
\Rightarrow \alpha_{\text{min}} &= \dfrac{\sigma_Y^2 - \sigma_{XY}}{\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}}
\end{aligned}
\]

## Problem 3: Validation Comparisons
### Part a)
$k$-fold cross-validation consists of splitting the training data into $k$ separate sets.

The data is then trainined on the $\frac{k-1}{k}$ fraction of the dataset. The remaining $k$th partition is used to validate and score the data. This is repeated with each distinct $k$th partition as the validation set and the rest as the training resulting in $k$ iterations and validation errors.

The final CV score is the average of these $k$ test scores.

### Part b)
#### i.
The single validation set approach results in a single score on the valdation set. This score can be highly dependent on the data selected in the train-test split and is not totally indicative of the true test error. It also reduces the total data that is used to test the data set. $k$-fold cross-validation addresses this by taking many small samples and, after all the iterations, trains the model on the entire dataset.

#### ii.
This is a subset of $k$-fold cross-validation where $k$ is the number of datapoints. It is highly computationally expensive for even a moderately sized dataset.

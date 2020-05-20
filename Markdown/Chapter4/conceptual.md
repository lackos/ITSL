# Chapter Four: Classification
# Conceptual Problems

## Problem One:
The derivation is as follows.
$$
\begin{aligned}
p(x) &= \frac{\exp(\beta_{0} + \beta_{1}x)}{1 + \exp(\beta_0 + \beta_{1}x)}\\
1 - p(x) &= 1 - \frac{\exp(\beta_0 + \beta_{1}x)}{1 + \exp(\beta_0 + \beta_{1}x)} \\
&= \frac{1 + \exp(\beta_0 + \beta_{1}x) - \exp(\beta_0 + \beta_{1}x)}{1 + \exp(\beta_0 + \beta_{1}x)} \\
&= \frac{1}{1 + \exp(\beta_0 + \beta_{1}x)}  \\
\Rightarrow \frac{p(x)}{1-p(x)}&= \frac{\exp(\beta_0 + \beta_{1}x)}{1 + \exp(\beta_0 + \beta_{1}x)} \times \frac{1 + \exp(\beta_0 + \beta_{1}x)}{1} \\
&= \exp(\beta_0 + \beta_{1}x)
\end{aligned}
$$

## Problem Two:
The posterior probability density of a point $y$ in class $k$ for $x$ in $X$ if given by (assuming normally distributed and common variance),
\[
p_{k} = \dfrac{\pi_{k}\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\dfrac{1}{2\sigma^2}\left(x - \mu_k\right)^2\right)}{\sum_{l=1}^K\pi_{l}\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\dfrac{1}{2\sigma^2}\left(x - \mu_l\right)^2\right)}.
\]

We will classify the point depending on the largest value of p_{k}. We can similfy this problem.  Firstly, the denominator is a constant normalization factor and therefore inconsequential and we are just left with the numerator. Taking the natural logarithm of both sides,

\[
\ln(p_k(x)) \propto \ln \left(\pi_{k}\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\dfrac{1}{2\sigma^2}\left(x - \mu_k\right)^2\right)\right).
\]

Also the logarithm is monotonic, therefore the larger $p_k(x)$ corresponds to the largest $\ln(p_k(x))$. We can further separate the logarithm,

\[
\begin{aligned}
\ln(p_k(x)) &\propto \ln\left(
\exp\left(-\dfrac{1}{2\sigma^2}\left(x - \mu_k\right)^2\right)
\right) + \ln\left(\pi_{k}\right) - \ln\left(\sqrt{2\pi\sigma^2}\right) \\
&= -\dfrac{1}{2\sigma^2}\left(x - \mu_k\right)^2
+ \ln\left(\pi_{k}\right) - \ln\left(\sqrt{2\pi\sigma^2}\right)
\end{aligned}
\]

The last term is again a constant of the distribution and identical for every classification. Expanding out the quadratic terms,

\[
\ln(p_k(x)) \propto -\dfrac{x^2}{2\sigma^2} - \dfrac{\mu_k}{2\sigma^2} + \dfrac{x\mu_k}{\sigma^2} + \ln\left(\pi_{k}\right)
\]

Lastly, we can drop of the $x^2$ factor. This may seem erroneous as $x$ is a variable, however, as it is independent of the class $k$ it will be a constant factor amongst all the class proabilities and therefore can be dropped.

## Problem Three

This is very similar to the derivation in problem two and therefore a lot will be omitted. All that is different here is $\sigma \rightarrow \sigma_k$, that it the probability densities do not have a common variance. Therefore we have,

\[
\begin{aligned}
\ln(p_k(x)) &\propto \ln\left(
\exp\left(-\dfrac{1}{2\sigma_k^2}\left(x - \mu_k\right)^2\right)
\right) + \ln\left(\pi_{k}\right) - \ln\left(\sqrt{2\pi\sigma_k^2}\right) \\
&= -\dfrac{1}{2\sigma_k^2}\left(x - \mu_k\right)^2
+ \ln\left(\pi_{k}\right) - \ln\left(\sqrt{2\pi\sigma_k^2}\right) \\
&=   -\dfrac{x^2}{2\sigma_k^2} - \dfrac{\mu_k}{2\sigma_k^2} + \dfrac{x\mu_k}{\sigma_k^2} + \ln\left(\pi_{k}\right) - \ln\left(\sqrt{2\pi\sigma_k^2}\right)
\end{aligned}
\]

We cannot dropped the $x^2$ term here as it has a class dependence, therefore there is a quadratic decision boundary. This can not be simplified further.

## Problem Five
### Part a)
*LDA will perform better on the test set*

### Part b)
*QDA will perform better on the test set*

### Part c)
*As the sample size of the training set increases, we expect QDA to perform better.*  This is due to the bias-variance tradeoff. A consequence of a small training set is an intrinsically larger variance. Therefore a model with low variance will perform better. As the training set increase and variance decreases reducing the bias with a more flexible QDA should give better results.

### Part d)
*False*. This is similar to fitting a non-linear regression to a linear relationship. It may perform well on the training set (overfit) but will poorly generalize to the test set (out of sample).

## Problem Six
This is a simple example of Logistic regression analysis. In terms of Log odds of getting an A, the regression is,
\[
\ln\left(\dfrac{p(X)}{1-p(X)}\right) = -6 + 0.05X_1 + X_2
\]
or in terms of probability,
\[
p(X) = \dfrac{\exp\left(-6 + 0.05X_1 + X_2\right)}{1 + \exp\left(-6 + 0.05X_1 + X_2\right)}.
\]
### Part a)
For $X_1 = 40$, $X_{2} = 3.5$ the probabilty of getting an A is,
\[
\begin{aligned}
p(X_1=40, X_2=3.5) &= \dfrac{\exp\left(-6 + 0.05\times 40 + 3.5\right)}{1 + \exp\left(-6 + 0.05\times 40 + 3.5\right)} \\
&= 0.3775
\end{aligned}
\]
Therefore there is a 38% chance of getting an A.

### Part b)
Using the Log odds approach, the hours needed for a 3.5 GPA student having an even chance of getting an A ($p(X) = 0.5$) is,
\[
\ln\left(\dfrac{0.5}{1-0.5}\right) = 0 = -6 + 0.05X_1 + 3.5 \\
\Rightarrow X_1 = 50
\]
THerefore this student will need to study 50 hours for an even chance of getting an A.

## Problem Seven
Bayes theorem of continuous distribution is given by,
\[
p(Y=k | X=x) = \dfrac{\pi_{k}\dfrac{1}{\sqrt{2\pi\sigma_k^2}}\exp\left(-\dfrac{1}{2\sigma_k^2}\left(x - \mu_k\right)^2\right)}{\sum_{l=1}^K\pi_{l}\dfrac{1}{\sqrt{2\pi\sigma_l^2}}\exp\left(-\dfrac{1}{2\sigma_l^2}\left(x - \mu_l\right)^2\right)}.
\]
Assuming that the variable is normally distributed with similar variance among classes ($\sigma_k = \sigma$), the probability density for the $k$th class is given by,
\[
f_{k}(x) = \dfrac{1}{\sqrt{2\pi \sigma^2}}\exp\left(-\dfrac{1}{2\sigma^2}\left(x-\mu_{k}\right)\right).
\]

In this problem we only have 2 classes ($k$ = yes ($y$) or no ($n$)) with  $\sigma^2 = 36$, $\pi_y = 0.8$, $\pi_n = 0.2$, $\mu_y=10$ and $\mu_n = 0$. In total, the probability of the company issuing dividends ($Y = y$) given percentage profit $X = 4$ can be mathematically written as,
\[
\begin{aligned}
p(Y=y | X=4) &=  \dfrac{\pi_{y}\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\dfrac{1}{2\sigma^2}\left(x - \mu_y\right)^2\right)}{\pi_{y}\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\dfrac{1}{2\sigma^2}\left(x - \mu_y\right)^2\right) + \pi_{n}\dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\dfrac{1}{2\sigma^2}\left(x - \mu_n\right)^2\right)} \\
&= 0.75
\end{aligned}
\]
Therefore there is a 75% chance of the company issuing dividends given that they experienced a percentage profit increase of 4.

## Problem Eight
Ignoring the problems of domain specificity in model evalution (raw error rate may not be the most important measure of performance) we can make a judgement on these models.

Definitionally, the KNN $k=1$ classifier will always perfectly classify the training set as it is perfectly flexible resulting in a training error rate of 0 (therefore we can infer the error rate of the test set is 36%). This means there is a worse test set performance of the KNN classifier. Also the logistic regression is a parametric model and gives interpretibility to the model which KNN does not (even using larger $k$ which may classify better than the logistic regression.)

This is once again a problem of bias-variance tradeoff.

## Problem Nine
### Part a)
\[
\begin{aligned}
p(x) &= \dfrac{odds}{odds + 1} \\
&= \dfrac{0.37}{0.37 + 0.37} \\
&= 27%
\end{aligned}
\]
### Part b)
\[
\begin{aligned}
odds &= \dfrac{p(x)}{1 - p(x)} \\
&= \dfrac{0.16}{1 1- 0.16} \\
&= \dfrac{4}{21}
\end{aligned}
\]

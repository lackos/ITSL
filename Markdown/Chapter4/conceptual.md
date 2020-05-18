# Chapter Four Conceptual Problems
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

This is very similar to the derivation in problem four and therefore a lot will be omitted. All that is different here is $\sigma \rightarrow \sigma_k$, that it the probability densities do not have a common variance. Therefore we have,

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

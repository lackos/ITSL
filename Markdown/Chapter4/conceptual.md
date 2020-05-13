# Chapter 4: Classification
## Problem 1:
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
## Problem 2:

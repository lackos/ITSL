# Chapter 7 Conceptual Problems
## Problem 1
The Cubic regression spline with a knot at $\xi$ has the form
\[
f(x) = \beta_{0} + \beta_{1}x + \beta_{2}x^{2} + \beta_{3}x^{3} + \beta_{4}\left(x-\xi\right)^{3}_{+}
\]
### Part a)
Find a choice of $a_{1}$, $b_{1}$, $c_{1}$ and $d_{1}$ for,

\[
f_{1}(x) = a_{1} + b_{1}x + c_{1}x^{2} + d_{1}x^{3}
\]

for $x<\xi$. Equating the two equations,

\[
a_{1} + b_{1}x + c_{1}x^{2} + d_{1}x^{3}= \beta_{0} + \beta_{1}x + \beta_{2}x^{2} + \beta_{3}x^{3}
\]

and equating coefficients we have,

\[
a_{1} = \beta_{0} \qquad  b_{1} = \beta_{1} \\
c_{1} = \beta_{2} \qquad  d_{1} = \beta_{3}. \\
\]

### Part b)
Find a choice of $a_{2}$, $b_{2}$, $c_{2}$ and $d_{2}$ for,

\[
f_{2}(x) = a_{2} + b_{2}x + c_{2}x^{2} + d_{2}x^{3}
\]

for $x \geq \xi$. Equating the two equations,

\[
a_{2} + b_{2}x + c_{2}x^{2} + d_{2}x^{3}= \beta_{0} + \beta_{1}x + \beta_{2}x^{2} + \beta_{3}x^{3} + \beta_{4}\left(x-\xi\right)^{3}
\]

Expanding the cubic term,

\[
\left(x-\xi\right)^{3} = x^3 -3\xi x^2 + 3\xi^{2} x - \xi^3.
\]

Substituting the expanded expression in the equation and comparing coefficients gives,

\[
a_{2} = \beta_{0} - \beta_{4}\xi^{3} \qquad  b_{2} = \beta_{1} + 3\beta{4}\xi^{2} \\
c_{2} = \beta_{2} - 3\beta_{4}\xi \qquad  d_{2} = \beta_{3} + \beta_{4}. \\
\]

### Part c)
Show that $f(x)$ is continuous at the knot.

\[
f_{1}(\xi) =  \beta_{0} + \beta_{1}\xi + \beta_{2}\xi^{2} + \beta_{3}\xi^{3}
\]

and

\[
\begin{aligned}
f_{2}(\xi) &= \beta_{0} - \beta_{4}\xi^{3} + \beta_{1}\xi + 3\beta_{4}\xi^{3} + \beta_{2}\xi^{2} - 3\beta_{4}\xi^{3} + \beta_{3}\xi^{3} + \beta_{4}\xi^{3} \\
&= \beta_{0} + \beta_{1}\xi + \beta_{2}\xi^{2} + \beta_{3}\xi^{3} + \left(-\xi^{3} +3\xi^{3} -3\xi^{3} + \xi^{3}\right)\beta_{4} \\
&=  \beta_{0} + \beta_{1}\xi + \beta_{2}\xi^{2} + \beta_{3}\xi^{3} \\
&= f_{1}(\xi)
\end{aligned}
\]

Therefore $f(x)$ is continous at the knot $\xi$ (and everywhere else as $f_{1}$ and $f_{2}$ are continuous functions).

### Part d)
Show that $f'(x)$ is continuous at the knot.
\[
f'_{1}(\xi) =  \beta_{1} + 2\beta_{2}\xi + 3\beta_{3}\xi^{2}
\]

and

\[
\begin{aligned}
f'_{2}(\xi) &= \beta_{1} + 3\beta_{4}\xi^{2} + 2\beta_{2}\xi - 6\beta_{4}\xi^{2} + 3\beta_{3}\xi^{2} + 3\beta_{4}\xi^{2} \\
&= \beta_{1} + 2\beta_{2}\xi + 3\beta_{3}\xi^{2} + \left(3\xi^{2} -6\xi^{2} +3\xi^{2} \right)\beta_{4} \\
&=   \beta_{1} + 2\beta_{2}\xi + 3\beta_{3}\xi^{2} \\
&= f'_{1}(\xi)
\end{aligned}
\]

Therefore the first derivative of $f$ is continuous at the knot and everywhere.

### Part d)
Show that $f''(x)$ is continuous at the knot.
\[
f''_{1}(\xi) =  2\beta_{2} + 6\beta_{3}\xi
\]

and

\[
\begin{aligned}
f''_{2}(\xi) &=  2\beta_{2} - 6\beta_{4}\xi + 6\beta_{3}\xi + 6\beta_{4}\xi \\
&= 2\beta_{2} + 6\beta_{3}\xi + \left(-6\xi + 6\xi \right)\beta_{4} \\
&= 2\beta_{2} + 6\beta_{3}\xi \\
&= f''_{1}(\xi)
\end{aligned}
\]

Therefore the second derivative of $f$ is continuous at the knot and everywhere.

## Problem 5
Considering the two smoothing functions,
\[
g_1 = \text{arg} \min_{g}\left(\sum_{i}\left(y_i - g_i(x)\right)^2 + \lambda\int\left(g^{(3)}(t)\right)dt\right)
 \\
g_2 = \text{arg} \min_{g}\left(\sum_{i}\left(y_i - g_i(x)\right)^2 + \lambda\int\left(g^{(4)}(t)\right)dt\right)
\]
### Part a)
For $\lambda \rightarrow \infty$ the effect of the constraint in maximized. In this case it reduces the amount of roughness of a function to the differential order of the constrained term. As $g_1$ has a lower order of smoothness constrained, it will have the lower training error than $g_2$.

## Part b)
It is not possible to definitively say which will have a greater test RSS as that depends on the variance and bias tradeoff. All we can say is that maximally constraining the $g_1$ will lower the variance and increase the bias of the model than maximally constraining $g_2$ will.

## Part c)
For $\lambda = 0$ there is no constraint and the training data is completely interpolated by the function and most definitely overfits the data and has a poor test RSS. Here $g_1 = g_2$.  

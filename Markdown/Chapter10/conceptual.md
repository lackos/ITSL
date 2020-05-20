# Chapter 10: Unsupervised Learning
# Conceptual Problems

## Problem One

### Part a)
Prove that
\[
\dfrac{1}{\left|C_k\right|}\sum_{i,i' \in C_{k}}\sum_{j=1}^{P} \left(x_{ij} - x_{i'j}\right)^2 = 2\sum_{i \in C_{k}}\sum_{j=1}^{P} \left(x_{ij} - \bar{x}_{kj}\right)^2
\]

where
\[
\bar{x}_{kj} = \dfrac{1}{\left|C_k\right|} \sum_{i \in C_{k}}x_{ij}.
\]

Also note that
\[
\sum_{i \in C_k}1 = \left|C_k\right|.
\]
Below is a detailed proof,

\[
\begin{aligned}
\dfrac{1}{\left|C_k\right|}\sum_{i,i' \in C_{k}}\sum_{j=1}^{P} \left(x_{ij} - x_{i'j}\right)^2 &= \sum_{j=1}^{P}\dfrac{1}{\left|C_k\right|}\sum_{i\in C_{k}}\sum_{i'\in C_{k}} \left(x_{ij} - x_{i'j}\right)^2 \\
&= \sum_{j=1}^{P}\dfrac{1}{\left|C_k\right|}\sum_{i\in C_{k}}\sum_{i'\in C_{k}} \left(x_{ij}^2 + x_{i'j}^2 -2x_{ij}x_{i'j}\right) \\
&= \sum_{j=1}^{P}\dfrac{1}{\left|C_k\right|} \left(\sum_{i\in C_{k}}\sum_{i'\in C_{k}}x_{ij}^2 + \sum_{i\in C_{k}}\sum_{i'\in C_{k}}x_{i'j}^2 -2\sum_{i\in C_{k}}\sum_{i'\in C_{k}}x_{ij}x_{i'j}\right) \\
&=   \sum_{j=1}^{P}\dfrac{1}{\left|C_k\right|} \left(\sum_{i\in C_{k}}x_{ij}^2\sum_{i'\in C_{k}}1 + \sum_{i'\in C_{k}}x_{i'j}^2\sum_{i\in C_{k}}1 -2\sum_{i\in C_{k}}x_{ij}\sum_{i'\in C_{k}}x_{i'j}\right) \\
&=  \sum_{j=1}^{P}\left(\sum_{i\in C_{k}}x_{ij}^2 + \sum_{i'\in C_{k}}x_{i'j}^2 -2\left|C_k\right|\bar{x}_{kj}^2\right) \\
&= 2\sum_{j=1}^{P}\left(\sum_{i\in C_{k}}x_{ij}^2 -\left|C_k\right|\bar{x}_{kj}^2\right)
\end{aligned}
\]

Now we can transform this by doing a type of "completing the square", and using $|C_k| = \dfrac{1}{\bar{x}_{kj}} \sum_{i \in C_{k}}x_{ij}$.

\[
\begin{aligned}
\dfrac{1}{\left|C_k\right|}\sum_{i,i' \in C_{k}}\sum_{j=1}^{P} \left(x_{ij} - x_{i'j}\right)^2 &= 2\sum_{j=1}^{P}\left(\sum_{i\in C_{k}}x_{ij}^2 -\left|C_k\right|\bar{x}_{kj}^2 - \left|C_k\right|\bar{x}_{kj}^2 + \left|C_k\right|\bar{x}_{kj}^2\right) \\
&= 2\sum_{j=1}^{P}\left(\sum_{i\in C_{k}}x_{ij}^2 -2\left|C_k\right|\bar{x}_{kj}^2  + \left|C_k\right|\bar{x}_{kj}^2\right) \\
&= 2\sum_{j=1}^{P}\left(\sum_{i\in C_{k}}x_{ij}^2 -2\sum_{i \in C_{k}}x_{ij}\bar{x}_{kj}  + \sum_{i \in C_k}\bar{x}_{kj}^2\right) \\
&= 2\sum_{j=1}^{P}\sum_{i\in C_{k}}\left(x_{ij}^2 -2x_{ij}\bar{x}_{kj}  + \bar{x}_{kj}^2\right) \\
&= 2\sum_{i \in C_{k}}\sum_{j=1}^{P} \left(x_{ij} - \bar{x}_{kj}\right)^2
\end{aligned}
\]

## Problem Two
### Part a)
\[
\{1\} \quad \{2\} \quad  \{3\} \quad \{4\} \quad \qquad \{1, 2\} \quad  \{3\} \quad \{4\} \quad \quad \{1, 2\} \quad  \{3,4\}\\
\begin{matrix}
\begin{pmatrix}
0 & 0.3 & 0.4 & 0.7 \\
0.3 & 0 & 0.5 & 0.8 \\
0.4 & 0.5 & 0 & 0.45 \\
0.7 & 0.8 & 0.45 & 0
\end{pmatrix}
\end{matrix} \rightarrow
\begin{pmatrix}
0 & 0.5 & 0.8 \\
0.5 & 0 & 0.45 \\
0.8 & 0.45 & 0
\end{pmatrix} \rightarrow
\begin{pmatrix}
0 & 0.8 \\
0.8 & 0
\end{pmatrix}
\]

### Part b)

\[
\{1\} \quad \{2\} \quad  \{3\} \quad \{4\} \quad \qquad \{1, 2\} \quad  \{3\} \quad \{4\} \quad \quad \{1, 2, 3\} \quad \{4\}\\
\begin{pmatrix}
0 & 0.3 & 0.4 & 0.7 \\
0.3 & 0 & 0.5 & 0.8 \\
0.4 & 0.5 & 0 & 0.45 \\
0.7 & 0.8 & 0.45 & 0
\end{pmatrix} \rightarrow
\begin{pmatrix}
0 & 0.4 & 0.7 \\
0.4 & 0 & 0.45 \\
0.7 & 0.45 & 0
\end{pmatrix} \rightarrow
\begin{pmatrix}
0 & 0.45 \\
0.45 & 0
\end{pmatrix}
\]

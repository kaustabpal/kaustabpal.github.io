+++
date = '2025-12-30T07:54:26+05:30'
draft = true
math = true
title = 'Linear Algebra'
url = '/linearalgebra'
+++

## Eigen values and eigen vectors

A matrix $A$ acts like a function. It takes in a vector $\boldsymbol{x}$ as input and gives out a vector $A\boldsymbol{x}$ as output. For all the vectors $\boldsymbol{x}$, if $A\boldsymbol{x}$ is a multiple of $\boldsymbol{x}$, i.e. $A\boldsymbol{x}=\lambda\boldsymbol{x}$, then $\boldsymbol{x}$ is called an eigen vector of $A$ and $\lambda$ is called an eigen value. Eigen vectors are the vectors for which $A\boldsymbol{x}$ is a scaled version of $\boldsymbol{x}$. For an $n \times n$ matrix, there will be $n$ eigen values. If the eigen values are all different, then each eigen values will give us a line of eigen vectors. If an eigen value is repeated, then we will get an entire plane of eigen vectors.

If the eigen value of a matrix $A$ is $0$, then the eigen vectors of the matrix makeup the null space of the matrix, i.e. $A\boldsymbol{x}=0\boldsymbol{x}=0 \implies A\boldsymbol{x}=0$.

If we have a projection matrix $P$ and $\boldsymbol{x}$ is a vector on the projection plane, then $P\boldsymbol{x}=\boldsymbol{x}$. So $\boldsymbol{x}$ is an eigen vector of $P$ with $\lambda=1$.

If $\boldsymbol{x}$ is perpendicular to the plane $P$, then $P\boldsymbol{x}=0$. So $\boldsymbol{x}$ is an eigen vector of $P$ with $\lambda=0$.

The eigen vectors for $\lambda=0$ fill up the null space of $P$.

The eigen vectors for $\lambda=1$ fill up the column space of $P$.

### Solving for eigen values and eigen vectors

We know

$$A\boldsymbol{x}=\lambda\boldsymbol{x} \implies (A-\lambda I)\boldsymbol{x}=0$$

To satisfy this, $(A-\lambda I)$ must be singular. Therefore, $\begin{vmatrix}A-\lambda I\end{vmatrix}=0$

Solving this, we will get $n$ solutions, i.e. $n$ eigenvalues. They maybe different or similar. If the eigenvalues are similar, then the eigen vectors cannot be uniquely determined. We have to choose them. For example, in the identity matrix, the eigen values are all $1$ but we can choose $n$ independent eigen vectors for the identity matrix.

Once we find the eigenvalues, we can use elimination to find the null space of $A-\lambda I$ for all $\lambda$. The vectors in the null space of $A-\lambda I$ are the eigen vectors of $A$ with eigen value $\lambda$.

Real symmetric matrices will always have real eigen values.

For anti-symmetric matrices i.e. $A^T=-A$, all eigen values are either zero or imaginary.

---

## Singular Value Decomposition

Singular value decomposition is the factorization of any $m \times n$ real or complex matrix. It is often referred to as the best factorization of a matrix. 

If $M$ is an $m\times n$ matrix, then it's singular decomposition form can be written as $M=U\Sigma V^T$. Here $U$ and $V$ are orthogonal matrices and $\Sigma$ is a diagonal matrix. 

The SVD is extremely useful in signal processing, least squares fitting of data, process control, etc.

In Singular value decomposition, a matrix $A\in R^{m \times n}$ is decomposed as $$A=U\Sigma V^T$$ where:

- $U\in R^{m\times m}$ contains orthonormal left singular vectors, which form an orthonormal basis for the column space of $A$. Any vector in the column space of $A$ can be expressed as a linear combination of the columns of $U$.

-  $V\in R^{n\times n}$ contains orthonormal left singular vectorscontains orthonormal right singular vectors, which form an orthonormal basis for the row space of 
A.

- $\Sigma\in R^{m\times n}$ is a diagonal matrix whose non-negative diagonal entries (singular values) scale the corresponding singular vectors.


We can also say that $V$ contains the eigen vectors of the correlation matrix $A^TA$ and $\Sigma$ contains the Eigen Values of $A^TA$. 

$$
A^TAV=V\Sigma^2
$$


Similarly $U$ contains the eigen vectors of $AA^T$ and $\Sigma$ contains the Eigen Values of $AA^T$.

$$
AA^TU=U\Sigma^2 
$$

The goal is to find the orthonormal basis vectors($V$) in the row space of $A$ in such a way that if $A$ acts as a linear transformation matrix, it would transform $V$ to some multiples of the orthonormal basis vectors($U$) in the column space of $A$. In the matrix form, we can write this as $AV=U\Sigma$ such that the columns of the $V$ matrix contains the orthonormal basis vectors in the row space of $A$, the columns of the $U$ matrix contains the orthonormal basis vectors in the column space of $A$ and $\Sigma$ is a diagonal matrix that contains the multiplier elements to the vectors in $U$.

We can write $AV=U \Sigma$ as $A=U\Sigma V^{-1} \implies A=U\Sigma V^T$.

Now we have to find the two orthogonal matrices $U$ and $V$.

To find $V$ we multiply $A^T$ on the left of both sides. Therefore we get

$$A^TA=V\Sigma ^T U^T U \Sigma V^T \implies A^TA=V\Sigma ^2 V^T$$

This implies that the columns of $V$ contains the eigen vectors of $A^TA$ and $\Sigma = \sqrt{\text{eigen values of }A^TA}$$.

Similarly, to find $U$ we multiply $A^T$ on the right of both sides. Therefore we get

$$AA^T=U\Sigma V^T V \Sigma ^T U^T \implies AA^T=U\Sigma ^2 U^T$$

This implies that the columns of $U$ contains the eigen vectors of $AA^T$ and $\Sigma = \sqrt{\text{eigen values of }AA^T}$$.

Now that we know $U$, $V$ and $\Sigma$, we can easily write $A$ as $A=U\Sigma V^T$.


### SVD Matrix Approximation

We can take a matrix $A$ and write it as three seperate matrices U Sigma and V

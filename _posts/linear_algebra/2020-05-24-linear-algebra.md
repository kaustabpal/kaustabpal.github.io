---
layout: post
title: "Linear Algebra"
tags: maths 
permalink: "linear_algebra"
excerpt: "An overview of the concepts of Linear Algebra."

---
{: class="table-of-content"}
* TOC
{:toc}

A linear equation is of the form $$Ax+By+Cz=-D$$

In the above equation, the variables $$A$$, $$B$$, $$C$$ and $$D$$ are constants and the variable $$x$$, $$y$$ and $$z$$ are called unknowns. When $$z$$ is $$0$$, the equation is in the 2-d space and represents a line. Otherwise the equation is in the 3-d space and represents a plane.

If we have a set of 2 or more linear equations, they are called a system of linear equations. If the lines/planes represented by those equations intersect at some points, then those points are called solutions to that system of equations.

A system of linear equations can have:
- an unique solution, i.e. the lines or planes intersect each other only at one point.
- an infinite number of solutions, i.e. the lines or planes overlap each other and thus have infinite points of intersection.
- no solution, i.e. the lines or planes do not intersect each other at all.

The fundamental problem of linear algebra is to find a solution to a system of linear equations.

Suppose we are given a system having two equations with two unknowns:

$$2x-y=0$$

$$-x+2y=3$$

In the matrix form, the system of equations can be written as

$$
\begin{bmatrix}
2 & -1 \\ -1 & 2 \end{bmatrix}
\begin{bmatrix}x \\ y \end{bmatrix}=\begin{bmatrix}0 \\ 3 \end{bmatrix}
$$

- Here the matrix $$A=\begin{bmatrix}2 & -1 \\ -1 & 2 \end{bmatrix}$$ is called the coefficient matrix.

- The vector $$\vec{x}=\begin{bmatrix}x \\ y \end{bmatrix}$$ is called the vector of unknowns.

- The vector $$\vec{b}=\begin{bmatrix}0 \\ 3 \end{bmatrix}$$ is the resulting vector.

- We can represent the above form generally as $$A\vec{x}=\vec{b}$$.

# Geometric point of view

 The above two equations represents two lines. If we can find a solution vector $$\vec{x}$$ such that it satisfies $$A\vec{x}=\vec{b}$$, then it means that the two lines intersect at the coordinates given by the vector $$\vec{x}$$. It might also be possible that we might have infinite solution vectors $$\vec{x}$$. This happens when the two lines overlap each other and therefore intersect at infinite points. If there does not exist a vector $$\vec{x}$$ which satisfies the equation $$A\vec{x}=\vec{b}$$ then we do not have a solution. This happens when the two lines are parallel to each other and therefore do not intersect.

# Transformation Matrix

 There is another way we can look at this. A matrix when multiplied to a vector gives us a new vector. Therefore we can say that the matrix acts like a function that takes a vector as an input and produces another vector as output. We will call this operation transformation and we will call the matrix a transformation matrix. 
 
 The output vector is just a [linear combination](#linear-combination) of the column vectors of the transformation matrix where the [coefficients of the linear combination](#linear-combination) are the values of the input vector, i.e. the first element of the input vector acts as the coefficient of the first column and the second element acts as the coefficient of the second column and so on. In our example, the matrix $$A$$ acts as a transformation matrix that transforms the unknown vector $$\vec{x}$$ to the vector $$\vec{b}$$. If there exists a vector $$\vec{x}$$ such that when it is transformed by $$A$$ we get the vector $$\vec{b}$$, then we say the system of equations have one solution. If such a vector doesn't exist, then there is no solution for the system of equations. If there exists an infinite number of vectors $$\vec{x}$$ which when transformed by the transformation matrix $$A$$ gives the vector $$\vec{b}$$, then we have an infinite number of solutions. 
 
 For every transformation matrix whose rows or columns are [linearly dependent](#linear-independence), there exists a vector $$\vec{b}$$ for which the system $$A\vec{x}=\vec{b}$$ have an infinite number of solutions.

# Inverse matrix

 Now that we have transformed the vector $$\vec{x}$$ to the vector $$\vec{b}$$ using the transformation matrix $$A$$, to go back from $$\vec{b}$$ to $$\vec{x}$$, we will use the [inverse matrix](#inverse-matrices) of the transformation matrix $$A$$ denoted as $$A^{-1}$$. To understand intuitively, whatever the transformation matrix does, the inverse matrix undoes. However not all transformation matrices have an inverse. When a transformation matrix have dependent rows or dependent columns then that transformation matrix does not have an inverse.

# Vector spaces

 The example we have been working with till now deals with vectors which are two dimensional. It means that the vectors have two components that represents the $$x$$ coordinates and $$y$$ coordinates of the two-dimensional plane. This two-dimensional plane is called a vector space. If we pick any two vectors from this two-dimensional plane, their linear combination will always lie in the plane. As a rule of thumb, if we take any number of vectors from a vector space, their linear combination will always lie in that vector space. Some other examples of vector spaces are the three-dimensional space, four-dimensional space and so on.

# Subspaces 

 From the two-dimensional space, if we pick a line that passes through the $$\begin{bmatrix}0 & 0\end{bmatrix}^T$$ vector, we will notice that this line acts like a vector space as-well. If we pick any vectors on this line we will see that their linear combination will always lie on this line. This line is called the subspace of the two-dimensional vector space.  A subspace is a vector space contained inside another vector space and must always contain the zero vector. A vector space can have multiple subspaces. 
 
 If we are given two subspaces $$P$$ and $$L$$. Then their union $$P \cup L$$ is not a subsapace but their intersection $$P \cap L$$ is a subspace. 
 
 Some examples of the possible subspaces of the three-dimensional vector space are:
- Any line through the zero vector $$\begin{bmatrix}0 & 0 & 0\end{bmatrix}^T$$.

- The whole three-dimensional vector space is also a subspace.

- Any plane through the zero vector $$\begin{bmatrix}0 & 0 & 0\end{bmatrix}^T$$.

- The single vector $$\begin{bmatrix}0 & 0 & 0\end{bmatrix}^T$$

# Span

 We are given two vectors $$\begin{bmatrix}0 & 1\end{bmatrix}^T$$ and $$\begin{bmatrix}1 & 0\end{bmatrix}^T$$. These two vectors lie on the two-dimensional vector space. If we pick any vector from this space, it will be a linear combination of these two vectors. That is why these two vectors are said to span the space. 
 
 Vectors $$\vec{v_1}, \vec{v_2}, \dots \vec{v_n}$$ are said to span a space when the space consists of all linear combinations of those vectors. If vectors $$\vec{v_1}, \vec{v_2}, \dots \vec{v_n}$$ span a space $$S$$, then $$S$$ is the smallest space containing those vectors.  

# Basis

 If we are given a sequence of vectors $$\vec{v_1}, \vec{v_2}, \dots \vec{v_n}$$ such that they are independent and they span a vector space, then these vectors are said to form the basis for that vector space. 
 
 Given a vector space, every basis of that space has the same number of vectors. That number is called the dimension of that space. There are exactly $$n$$ vectors in every basis for the n-dimensional space.

# Four fundamental subspaces

![Four fundamental subspaces](assets/img/bigPic.png)
*Fig 1: The four fundamental subspaces. (Image source: Sec 3.6 Strang, Gilbert. Introduction to Linear Algebra, 2009.)*

## Column Space

The column space of a matrix is all the linear combinations of the columns of the matrix.

Suppose we have a matrix $$A=\begin{bmatrix} 1 & 3\\ 2 & 3\\ 4 & 1\end{bmatrix}$$.

The column space of A will contain all the linear combinations of the vectors $$\begin{bmatrix}1 & 2 & 4\end{bmatrix}^T$$ and $$\begin{bmatrix}3 & 3 & 1\end{bmatrix}^T$$.

To understand the column space in terms of the equation $$A\vec{x}=b$$, $$b$$ is the linear combination of the columns of the matrix $$A$$. Therefore, $$b$$ has to be in the column space produced by $$A$$ for the system to have a solution $$\vec{x}$$ else $$A\vec{x}=b$$ is unsolvable. The column space of a matrix $$A$$ is represented as $$C(A)$$. For a matrix $$A$$ with m rows and n columns, the column space of $$A$$ will be a subspace of the vector space $$R^m$$.

## Null Space

The null space of a matrix $$A$$ contains all the solution vectors $$\vec{x}$$ to the equation $$A\vec{x}=0$$. $$A\vec{x}$$ is the linear combination of the column vectors of $$A$$. When the null space of $$A$$ contains only the zero vector, then the column vectors of $$A$$ are said to be independent. For a matrix $$A$$ with $$m$$ rows and $$n$$ columns, the null space of $$A$$ will be a subspace of the vector space $$R^n$$.

## Row Space

The row space of a matrix is all the linear combinations of the rows of the matrix. For the matrix $$A=\begin{bmatrix} 1 & 3\\ 2 & 3\\ 4 & 1\end{bmatrix}$$, the row space will contain all the linear combinations of the rows $$\begin{bmatrix} 1 & 3\end{bmatrix}$$, $$\begin{bmatrix} 2 & 3\end{bmatrix}$$ and $$\begin{bmatrix} 4 & 1\end{bmatrix}$$  of the matrix. The row space of a matrix $$A$$ is represented as $$C(A^T)$$ because the rows of the matrix $$A$$ will become the columns of the matrix $$A^T$$. Therefore we can say that the row space of $$A$$ is the column space of $$A^T$$. For a matrix $$A$$ with m rows and n columns, the row space of $$A$$ will be a subspace of the vector space $$R^n$$.

## Left Null Space

This is also called the null space of $$A^T$$. The null space of the matrix $$A^T$$ contains all the solutions $$\vec{y}$$ to the equation $$A^Ty=0$$. $$A^Ty$$ is the linear combination of the row vectors of $$A$$ or the column vectors of $$A^T$$. For a matrix $$A$$ with m rows and n columns, $$A^T$$ will have n rows and m columns. The null space of $$A^T$$ will be a subspace of the vector space $$R^m$$.

# Orthogonality

Two vectors $$\vec{x}$$ and $$\vec{y}$$ are said to be orthogonal if $$\vec{x}^T\vec{y}=0$$. Orthogonal means perpendicular, i.e. the vectors $$\vec{x}$$ and $$\vec{y}$$ are perpendicular to each other. All vectors are orthogonal to the zero vector.

Two subspaces $$S$$ and $$T$$ are said to be orthogonal to each other if all the vectors in subspace $$S$$ are orthogonal to all the vectors in subspace $$T$$. For example, the row space of a matrix is orthogonal to the nullspace of that matrix and the column space of that matrix is orthogonal to the left null space of that matrix.

# Projections

The equation $$A\vec{x}=\vec{b}$$ will have a solution only when $$\vec{b}$$ lies on the column space of $$A$$. However sometimes, due to measurement error, $$\vec{b}$$ might not lie on the column space of $$A$$ and therefore $$A\vec{x}=\vec{b}$$ will have no solution. So to find the best possible solution, we project vector $$\vec{b}$$ onto a vector $$\vec{p}$$. The vector $$\vec{p}$$ lies on the column space of $$A\vec{x}$$. We then solve for $$A\vec{\hat{x}}=\vec{p}$$ where $$\vec{\hat{x}}$$ is the best possible solution for $$A\vec{x}=\vec{b}$$.

![Projection](/assets/img/projection.jpg)
*Fig 2: Projection of vector b on vector a.*

If we have a vector $$\vec{b}$$ and a line determined by a vector $$\vec{a}$$, we do the projection of $$\vec{b}$$ on line $$\vec{a}$$ to find the vector $$\vec{p}$$.

Here, the vector $$\vec{p}$$ is a scalar multiple of vector $$\vec{a}$$.

$$\vec{p}=x\vec{a}$$

Now from the figure we can see that the vector $$\vec{a}$$ is orthogonal to the vector $$\vec{e}$$. Therefore

$$\vec{a^T}\vec{e}=0$$

$$\implies \vec{a^T}(\vec{b}-\vec{p})=0$$

$$\implies \vec{a^T}(\vec{b}-x\vec{a})=0$$

$$\implies x\vec{a^T}\vec{a}=\vec{a^T}\vec{b}$$

$$\implies
x=\dfrac{\vec{a^T}\vec{b}}{\vec{a^T}\vec{a}}$$

Therefore we can write

$$\vec{p}=\vec{a}x \implies \vec{p}=\dfrac{\vec{a}\vec{a^T}\vec{b}}{\vec{a^T}\vec{a}}$$

## Projection matrix

Now we want to write this projection in terms of a projection matrix $$P$$ such that $$\vec{p}=P\vec{b}$$.

Therefore we can write the projection matrix $$P$$ as

$$P=\dfrac{\vec{a}\vec{a^T}}{\vec{a^T}\vec{a}}$$

The column space of the matrix $$P$$ is spanned by the line $$\vec{a}$$ because for any $$\vec{b}$$, $$P\vec{b}$$ lies on $$\vec{a}$$ which means $$\vec{a}$$ is the linear combination of the columns of the matrix $$P$$.

The projection matrix $$P$$ is a [symmetric matrix](#symmetric-matrices) which means $$P^2\vec{b}=P\vec{b}$$ because the projection of the vector already on the line $$\vec{a}$$ will be the same vector only.

# Orthonormal vectors

For a matrix $$A$$, it's columns are guaranteed to be independent if they are othonormal vectors. Vectors $$q_1$$, $$q_2$$, $$\dots$$ $$q_n$$ are said to be orthonormal if

$$q_i^Tq_j= \begin{cases}
  0 & \text{if i}\neq \text{j}\\    
  1 & \text{if i = j}    
\end{cases}$$

In other words, the vectors all have unit length and are perpendicular to each other. Orthonormal vectors are always independent.

## Orthonormal Matrix

A matrix $$Q$$ whose columns are orthonormal vectors is called an orthonormal matrix.
A square orthonormal matrix is called an orthogonal matrix. If $$Q$$ is square then

$$Q^TQ=I$$ 

$$\implies Q^T=Q^{-1}$$

Suppose we have a matrix $$A=\begin{bmatrix}1 & 1\\ 1 & -1\end{bmatrix}$$.
This is not an orthogonal matrix. We can adjust the matrix to make it an orthogonal matrix. Since the length of the column vectors were $$\sqrt{2}$$, we divide the matrix by $$\sqrt{2}$$ to make the columns as unit vectors.

We get the orthogonal matrix as $$Q=\dfrac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\1 & -1\end{bmatrix}$$.

We convert the orthogonal matrix $$A$$ to $$Q$$ because the matrix that projects onto the columnspace of $$Q$$ is
$$P=Q(Q^TQ)^{-1}Q^T$$
We know $$Q^TQ=I$$. Therefore $$P=QQ^T$$. If $$Q$$ is square then $$P=I$$ because the columns of $$Q$$ span the entire space.

If the columns of $$Q$$ are orthonormal then $$\hat{x}=Q^Tb$$.

# Determinant of a matrix

The determinant is a number associated with a square matrix. If $$A$$ is a square matrix, then it's column vectors form the edges of a box. The determinant of the matrix $$A$$ is the volumn of the box. The determinant is represented by $$\begin{vmatrix}A\end{vmatrix}$$.

If $$T$$ is a transformation matrix and it transforms $$A$$ to $$U$$, then $$\begin{vmatrix}T\end{vmatrix}$$ tells us by how much times the volumn of $$A$$ will change when it will be transformed to $$U$$.

The determinant of a singular matrix is always $$0$$.

# Eigen values and eigen vectors

A matrix $$A$$ acts like a function. It takes in a vector $$\vec{x}$$ as input and gives out a vector $$A\vec{x}$$ as output. If $$A\vec{x}$$ is a multiple of the vector $$\vec{x}$$, i.e. $$A\vec{x}=\lambda\vec{x}$$, then $$\vec{x}$$ is called an eigen vector of $$A$$ and $$\lambda$$ is called an eigen value. Eigen vectors are the vectors for which $$A\vec{x}$$ is a scaled version of $$\vec{x}$$. For an $$n \times n$$ matrix, there will be $$n$$ eigen values.

If the eigen value of a matrix $$A$$ is $$0$$, then the eigen vectors of the matrix makeup the null space of the matrix, i.e. $$A\vec{x}=0\vec{x}=0 \implies A\vec{x}=0$$.

If we have a projection matrix $$P$$ and $$\vec{x}$$ is a vector on the projection plane, then $$P\vec{x}=\vec{x}$$. So $$\vec{x}$$ is an eigen vector of $$P$$ with $$\lambda=1$$.

If $$\vec{x}$$ is perpendicular to the plane $$P$$, then $$P\vec{x}=0$$. So $$\vec{x}$$ is an eigen vector of $$P$$ with $$\lambda=0$$.

The eigen vectors for $$\lambda=0$$ fill up the null space of $$P$$.

The eigen vectors for $$\lambda=1$$ fill up the column space of $$P$$.

## Solving for eigen values and eigen vectors

We know

$$A\vec{x}=\lambda\vec{x} \implies (A-\lambda I)\vec{x}=0$$

To satisfy this, $$(A-\lambda I)$$ must be singular. Therefore,

$$\begin{vmatrix}A-\lambda I\end{vmatrix}=0$$

Solving this, we will get $$n$$ solutions, i.e. $$n$$ eigenvalues. They maybe different or similar. If the eigenvalues are similar, then the eigen vectors cannot be uniquely determined. We have to choose them. For example, in the identity matrix, the eigen values are all $$1$$ but we can choose $$n$$ independent eigen vectors for the identity matrix.

Once we find the eigenvalues, we can use elimination to find the null space of $$A-\lambda I$$ for all $$\lambda$$. The vectors in the null space of $$A-\lambda I$$ are the eigen vectors of $$A$$ with eigen value $$\lambda$$.

Real symmetric matrices will always have real eigen values.

For anti-symmetric matrices i.e. $$A^T=-A$$, all eigen values are either zero or imaginary.

## Diagonalizing a matrix

If $$A$$ has $$n$$ linearly independent eigen vectors, then we can put those eigen vectors in the columns of a matrix $$S$$. Thus we get

$$AS=A\begin{bmatrix}x_1 & x_2 & \dots & x_n\end{bmatrix}$$

$$\implies AS= \begin{bmatrix}\lambda_1x_1 & \lambda_2x_2 & \dots & \lambda_nx_n\end{bmatrix}$$

$$\implies AS= S\begin{bmatrix}
\lambda_1 & 0 & \dots & 0\\
0 & \lambda_2 & & 0 \\
\vdots & & \ddots & \vdots \\
0 & \dots & 0 & \lambda_n
\end{bmatrix} = S\Lambda$$

Now, since the columns of $$S$$ are independent, we know that $$S^{-1}$$ exists. Therefore we can multiply both sides of $$AS=S\Lambda$$ by $$S^{-1}$$ and thus get

$$S^{-1}AS=\Lambda \implies A=S\Lambda S^{-1}$$

Here $$\Lambda$$ is a diagonal matrix whose non-zero entries are the eigenvalues of $$A$$. This form of writing the matrix $$A$$ is called diagonalization and it is important as it will simplifies calculation.

One of the applications of diagonalization is finding $$A^k$$ easily.

We know $$Ax=\lambda x$$

Therefore we can write $$A^2 x = \lambda Ax = \lambda^2x$$.

Now we can write $$A$$ as $$A=S\Lambda S^{-1}$$. Therefore

$$A^2=S\Lambda S^{-1}S\Lambda S^{-1}=S\Lambda^2 S^{-1}$$

Similarly $$A^k= S^{-1}=S\Lambda^k S^{-1}$$. Thus we can easily find $$A^k$$.

# Symmetric matrices

Symmetric matrices are one of the most special matrices in linear algebra. For a matrix $$A$$, if $$A=A^T$$ then $$A$$ is called a symmetric matrix. Symmetric matrices have some properties which makes them so special. The important properties of symmetric matrices are:

1. The eigen values of real symmetric matrices are always real.

2. If the real symmetric matrix has distinct eigen values then it's eigen vectors are always orthonormal.

3. Real symmetric matrices are always diagonalizable.

4. For a symmetric matrix, the eigen values have the same sign as the pivots of the matrix.

# Positive definite marices

Positive definite matrices are another type of special matrices in linear algebra. A matrix $$A$$ is said to be positive definite if $$x^TAx>0$$ for every non-zero vector $$x$$. Positive definite matrices have some special properties:

1. All eigen values in a positive definite matrix are positive.

2. The pivots of the matrix are positive.

3. All sub-determinants are positive.

4. If $$A$$ is a positive definite matrix, then $$A^{-1}$$ is also positive definite.

5. If $$A$$ and $$B$$ are positive definite matrices, then $$A+B$$ is also positive definite matrix.

6. If $$A$$ is a rectangular $$m\times n$$ matrix, then $$A^TA$$ is a square, symmetric and positive definite matrix.

A quadratic real function $$f(x)$$ on $$n$$ real variables can be written as $$x^TAx$$. If

$$x=\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}$$ and $$A=\begin{bmatrix}a & b \\
b & c \end{bmatrix}$$, then $$x^TAx=ax_1^2+2bx_1x_2+cx_2^2$$. 

The matrix $$A$$ being positive definite means that the function $$f(x)$$ has a unique minimum at $$x=0$$ and is strictly positive for all other $$x's$$.

# Similar Matrices

Two square matrices $$A$$ and $$B$$ are said to be similar if $$B=M^{-1}AM$$ for some matrix $$M$$. This allows us to put matrices in a family such that all matrices in a family are similar to each other. Each family is represented by a diagonal matrix.

If $$A$$ has a full set of eigen vectors, we can create its eigen vector matrix $$S$$ and write $$S^{-1}AS=\Lambda$$. So $$A$$ is similar to $$\Lambda$$ it's eigen values matrix. Similarly for the same $$A$$ if $$M^{-1}AM=B$$, then $$A$$ is also similar to $$B$$. Similar matrices have the same eigen values.

If two matrices have the same distinct eigen values then they are similar. However if two matrices have the same repeated eigen values then they may not be similar.

# Singular value decomposition

Singular value decomposition is the factorization of any $$m \times n$$ real or complex matrix. It is often referred to as the best factorization of a matrix. If $$M$$ is an $$m\times n$$ matrix, then it's singular decomposition form can be written as $$M=U\Sigma V^T$$. Here $$U$$ is an orthogonal matrix, $$\Sigma$$ is a diagonal matrix and $$V$$ is an orthogonal matrix. The SVD is extremely useful in signal processing, least squares fitting of data, process control, etc.

In Singular value decomposition, the goal is to find the orthonormal basis vectors($$V$$) in the row space of $$A$$ in such a way that if $$A$$ acts as a linear transformation matrix, it would transform $$V$$ to some multiples of the orthonormal basis vectors($$U$$) in the column space of $$A$$. In the matrix form, we can write this as $$AV=U\Sigma$$ such that the columns of the $$V$$ matrix contains the orthonormal basis vectors in the row space of $$A$$, the columns of the $$U$$ matrix contains the orthonormal basis vectors in the column space of $$A$$ and $$\Sigma$$ is a diagonal matrix that contains the multiplier elements to the vectors in $$U$$.

We can write $$AV=U \Sigma$$ as $$A=U\Sigma V^{-1} \implies A=U\Sigma V^T$$.

Now we have to find the two orthogonal matrices $$U$$ and $$V$$.

To find $$V$$ we multiply $$A^T$$ on the left of both sides. Therefore we get

$$A^TA=V\Sigma ^T U^T U \Sigma V^T \implies A^TA=V\Sigma ^2 V^T$$

This implies that the columns of $$V$$ contains the eigen vectors of $$A^TA$$ and $$\Sigma = \sqrt{\text{eigen values of }A^TA}$$.

Similarly, to find $$U$$ we multiply $$A^T$$ on the right of both sides. Therefore we get

$$AA^T=U\Sigma V^T V \Sigma ^T U^T \implies AA^T=U\Sigma ^2 U^T$$

This implies that the columns of $$U$$ contains the eigen vectors of $$AA^T$$ and $$\Sigma = \sqrt{\text{eigen values of }AA^T}$$.

Now that we know $$U$$, $$V$$ and $$\Sigma$$, we can easily write $$A$$ as $$A=U\Sigma V^T$$.

<br/>

---

<br/>

# Appendix: Definitions and Proofs

## Linear combination

Given two vectors $$\vec{v_1}$$ and $$\vec{v_2}$$ and scalars $$C$$ and $$D$$, the sum $$C\vec{v_1}+D\vec{v_2}$$ is called a linear combination of the vectors $$\vec{v_1}$$ and $$\vec{v_2}$$. The scalars $$C$$ and $$D$$ are called the coefficients of the linear combination. If vectors $$\vec{v_1}$$ and $$\vec{v_2}$$ are column vectors of a matrix $$A$$ and vector $$\vec{x}=\begin{bmatrix}C & D\end{bmatrix}^T$$, then $$A\vec{x}$$ gives us the linear combination of the column vectors of the matrix $$A$$ where the coefficients of the linear combination are the values of the vector $$\vec{x}$$.

## Linear Independence

Vectors $$\vec{v_1}, \vec{v_2}, \dots \vec{v_n}$$ are called linearly independent if the linear combination of these vectors is $$0$$ only if the coefficients of the linear combination are all $$0$$. If the linear combination of these vectors are $$0$$ for coefficients which are not all $$0$$, then the vectors are said to be dependent. If the vectors are column vectors of a matrix $$A$$, then the columns of the matrix are said to be independent if $$A\vec{x}=0$$ only when $$\vec{x}=0$$. Geometrically, two vectors are said to be independent if they do not lie on the same line. Three vectors are independent if they do not lie on the same plane.

## Elimination with Matrices

To solve a system of linear equations we need to solve for the unknowns. Elimination is the technique most commonly used by computer softwares to solve systems of linear equations. It finds a solution $$\vec{x}$$ to $$A\vec{x}=b$$ whenever $$A$$ is [invertible](#inverse-matrices).

For explaining the elimination technique, let us take an example. Suppose we have a system of equations as

$$x+2y+z=2$$

$$3x+8y+z=12$$

$$4y+z=2$$

In this example, we have

$$A=\begin{bmatrix}1 & 2 & 1\\ 3 & 8 & 1\\ 0 & 4 & 1\end{bmatrix}$$

$$b=\begin{bmatrix}2\\ 12\\ 2\end{bmatrix}$$

$$x=\begin{bmatrix}x\\ y\\ z\end{bmatrix}$$

Our goal is to perform operations on matrix $$A$$ and transform the matrix into an echelon form $$U$$. The operations that will be done on matrix $$A$$ will also be done on the vector $$b$$. The reason we want to transform matrix $$A$$ to $$U$$ is because it becomes easier to solve for the unknowns using back substitution.

The number $$1$$ in the upper left corner of matrix $$A$$ is called the first pivot. We multiply the first row with an appropriate value which in this case is $$3$$ and subtract it from the second row. The first number in the second row now becomes $$0$$ and the matrix becomes

$$\begin{bmatrix}1 & 2 & 1\\ 0 & -2 & 2\\ 0 & 4 & 1\end{bmatrix}$$

The number $$-2$$ in the second row now becomes the second pivot. We now multiply the second row by the appropriate value which in this case is $$2$$ and add it to the third row. The first number in the third row was already $$0$$ but after this operation the second number in the third row also becomes $$0$$ and we achieve an upper triangular matrix. The number $$5$$ in the third row is called the third pivot.

$$\begin{bmatrix}1 & 2 & 1\\ 0 & -2 & 2\\ 0 & 0 & 5\end{bmatrix}$$

The whole purpose of elimination is to go from $$A$$ to $$U$$. The operations that we performed here to transform $$A$$ to $$U$$ is only specific to our current matrix $$A$$ and it will vary depending on the values of the matrix.

While we did all these operations on $$A$$, we also need to do those same operations on vector $$b$$. Thus after performing the operations on vector $$b$$ we get the new vector $$c$$ as

$$c=\begin{bmatrix}2\\ 6\\ -10\end{bmatrix}$$

After elimination, our new system of equations becomes

$$\begin{bmatrix}1 & 2 & 1\\ 0 & -2 & 2\\ 0 & 0 & 5\end{bmatrix}\begin{bmatrix}x\\ y\\ z\end{bmatrix}=\begin{bmatrix}2\\ 6\\ -10\end{bmatrix}$$

which can also be written as

$$x+2y+z=2$$

$$-2y+2z=6$$

$$5z=-10$$

Performing back substitution on these equations, we can easily get the solution of the system as

$$\begin{bmatrix}x\\ y\\ z\end{bmatrix}=\begin{bmatrix}2 \\ 1 \\ -2 \end{bmatrix}$$

## Rank of a matrix

Given a matrix $$A$$, the rank of the matrix is equivalent to the number of pivots columns in the matrix. The rank of the matrix tells us the number of independent rows or columns in the matrix. The rank of the matrix is equal to the rank of the transpose of the matrix.

## Inverse Matrices

Suppose we have a square matrix $$E$$. If there exists a matrix which when multiplied with the matrix $$E$$ gives an Identity Matrix $$I$$, then that matrix is called the inverse matrix of $$E$$ and is represented by $$E^{-1}$$.

Not all matrices have an inverse. A singular matrix i.e. a matrix whose determinant is 0 has no inverse and there will always exist a non-zero vector $$\vec{x}$$ which when multiplied with that matrix will result in a zero vector.

The inverse of the product of two matrices is

$$(AB)^{-1}=B^{-1}A^{-1}$$

We can use the Gauss-Jordan method to find the inverse of a matrix.

Suppose we have a matrix $$A=\begin{bmatrix}1 & 3\\ 2 & 7\end{bmatrix}$$. Let $$A^{-1}=\begin{bmatrix}a & b\\ c & d\end{bmatrix}$$.

We know that $$AA^{-1}=I$$. Therefore we can say

$$A=\begin{bmatrix}1 & 3\\ 2 & 7\end{bmatrix}\begin{bmatrix}a & b\\ c & d\end{bmatrix}=\begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix}$$

To solve this, we can write in the form

$$\left[ \begin{array}{cc|cc} 1 & 3 & 1 & 0 \\ 2 & 7 & 0 & 1 \end{array} \right]$$

This is called an augmented matrix. Here the left-hand side of $$ \mid $$ represents matrix $$A$$ and the right-hand side of $$\mid$$ represents the identity matrix $$ I $$. To find the pivots on the side representing $$A$$ we use elimination and get

$$\left[ \begin{array}{cc|cc} 1 & 3 & 1 & 0 \\ 0 & 1 & -2 & 1 \end{array} \right]$$

Since the left-hand side is in upper triangular form, we now use elimination upwards to get

$$\left[ \begin{array}{cc|cc} 1 & 0 & 7 & -3 \\ 0 & 1 & -2 & 1 \end{array} \right]$$

If we write the operations that we performed in the form of a transformation matrix $$E$$, then we can write $$E[A\mid I]=[I\mid E]$$. Since $$EA$$ produces $$I$$, therefore we can say $$E$$ is $$A^{-1}$$ and thus we solve for $$A^{-1}$$.



## Solving $$Ax=0$$ and special solutions

Let $$A=\begin{bmatrix}
1 & 2 & 2 & 2\\
2 & 4 & 6 & 8\\
3 & 6 & 8 & 10\end{bmatrix}
$$. If we reduce $$A$$ to it's echelon form $$U$$,

we will have $$
U=\begin{bmatrix}
1 & 2 & 2 & 2\\
0 & 0 & 2 & 4\\
0 & 0 & 0 & 0\end{bmatrix}
$$

Here the first pivot is the first element of the first column i.e. $$1$$. We don’t find a pivot in the second column, so our next pivot is the $$2$$ in the
third column of the second row. The first and the third columns are called the **pivot columns** as they contain the pivots, and the variables which form the pivot elements in these pivot columns are called the **pivot variables**. The remaining columns are called **free columns**.

In our matrix $$A$$, columns $$1$$ and $$3$$ are the pivot columns while columns $$2$$ and $$4$$ are the free columns. Therefore in our solution vector $$x=\begin{bmatrix}
x_1 & x_2 & x_3 & x_4
\end{bmatrix}^T$$, we can assign any value to $$x_2$$ and $$x_4$$ as these variables multiply with the free columns. These variables are called **free variables**.
Suppose we assign $$x_2=1$$ and $$x_4=0$$.

Then $$2x_3+4x_4=0 \implies x_3=0$$

and $$x_1+2x_2+2x_3+2x_4=0 \implies x_1=-2$$

Thus one of the solution vectors is
$$x=\begin{bmatrix}
-2 & 1 & 0 & 0\end{bmatrix}^T$$. The solution vector means that $$-2$$ times the first column added with the second column will give us the zero vector. Therefore any multiple of the solution vector $$x$$ will also give the zero vector and thus a solution.

Similarly if we assign $$x_2=0$$ and $$x_4=1$$. Then we get the solution vector as:
$$x=\begin{bmatrix}
2 & 0 & -2 & 1\end{bmatrix}^T$$. This solution vector means that $$2$$ times the first column added with $$-2$$ times the third column added with the fourth column will give us the zero vector. Also any multiple of this solution vector will also give us a zero vector and thus a solution.
The two solution vectors that we got are called **special solutions**. The solution $$x$$ is the collection of all the linear combinations of these special solution vectors.

The number of free columns in the matrix is the rank of the matrix subtracted from the number of columns of the matrix ($$n-r$$). The number of special solutions is equal to the number of free columns of the matrix. The solution $$x$$ for the equation $$Ax=0$$ is the collection of the linear combination of all the special solutions of the matrix. These solutions for the equation $$Ax=0$$ form the null space of the matrix $$A$$.

## Projection matrix in high dimensional spaces

Here we will see what the projection matrix will be like in a higher dimensional space like $$R^3$$.

Suppose in $$R^3$$, we want to project a vector $$\vec{b}$$ onto a closed point $$p$$ on a plane. If $$\vec{a_1}$$ and $$\vec{a_2}$$ form the basis for the plane, then the plane is the columnspace of the matrix $$\begin{bmatrix}a_1 & a_2\end{bmatrix}$$. For the projection vector of $$\vec{b}$$, $$\vec{p}$$ must be a linear combination of $$\vec{a_1}$$ and $$\vec{a_2}$$ for it to lie on the plane. Therefore we can write

$$\hat{x_1}\vec{a_1}+\hat{x_2}\vec{a_2}=\vec{p}$$

$$\implies \begin{bmatrix} \vec{a_1} & \vec{a_2}\end{bmatrix}\begin{bmatrix}\hat{x_1}\\ \hat{x_2}\end{bmatrix}=\vec{p}$$

$$\implies A\hat{x}=\vec{p}$$

If $$\vec{e}=\vec{b}-\vec{p}$$ and $$\vec{e}$$ is orthogonal to the plane, then $$\vec{e}$$ is orthogonal to any vector on the plane. Therefore

$$\vec{a_1^T}\vec{e}=0 \quad and \quad \vec{a_2^T}\vec{e}=0 \\ \implies \vec{a_1^T}(\vec{b}-\vec{p})=0 \quad and \quad \vec{a_2^T}(\vec{b}-\vec{p})=0 \\ \implies \vec{a_1^T}(\vec{b}-A\hat{x})=0 \quad and \quad \vec{a_2^T}(\vec{b}-A\hat{x})=0 $$

In matrix form we can write

$$A^T(\vec{b}-A\hat{x})=0$$ 

$$\implies A^TA\hat{x}=A^T\vec{b}$$

Multiplying both sides by $$(A^TA^{-1})$$, we have:

$$\hat{x}=(A^TA)^{-1}A^T\vec{b}$$

Now $$\vec{p}=A\hat{x}$$. Therefore

$$\vec{p}=A(A^TA)^{-1}A^T\vec{b}$$

and the projection matrix $$P=A(A^TA)^{-1}A^T$$.

Now if $$A$$ is a square invertible matrix then it's column space would be the entire space and $$\vec{b}$$ would lie in the space and $$P$$ matrix would be equal to the identity matrix $$I$$. If $$\vec{b}$$ lies in the column space, then $$P\vec{b}=\vec{b}$$. If $$\vec{b}$$ lies perpendicular to the column space, then $$P\vec{b}=0$$ which means $$\vec{b}$$ lies in the left nullspace of $$A$$. This means $$A\hat{x}=P\vec{b}=0 \implies A\hat{x}=0$$.

A typical vector $$\vec{b}$$ will have it's projection $$\vec{p}$$ in the column space of $$A$$, and the component $$\vec{e}=\vec{b}-\vec{p}$$ is perpendicular to the column space of $$A$$. This $$\vec{e}$$ lies in the left null space of $$A$$ as the column space is perpendicular to the left null space.

## How to make a matrix orthonormal?

We start with two independent vectors $$\vec{a}$$ and $$\vec{b}$$ and want to find the orthonormal vectors $$\vec{q_1}$$ and $$\vec{q_2}$$.
![Orthogonal](/assets/img/orthogonal.jpg)

First we get two vectors $$\vec{A}$$ and $$\vec{B}$$ such that they are orthogonal to each other.

Let $$\vec{A}=\vec{a}$$. For $$\vec{B}$$ to be orthogonal to $$\vec{A}$$ it needs to be in the space spanned by $$\vec{a}$$ and $$\vec{b}$$ by projecting $$\vec{b}$$ onto $$\vec{a}$$ and letting $$\vec{B} = \vec{b} − \vec{p}$$. ($$\vec{B}$$ is what we previously called $$\vec{e}$$). Therefore we can write

$$\vec{B}=\vec{b}-\dfrac{\vec{A}^T\vec{b}}{\vec{A}^T\vec{A}}\vec{A}$$

Now the orthonormal vectors $$\vec{q_1}=\dfrac{\vec{A}}{\\|\vec{A}\\|}$$ and $$\vec{q_2}=\dfrac{\vec{B}}{\\|\vec{B}\\|}$$.

If we started with three vectors $$\vec{a}$$, $$\vec{b}$$ and $$\vec{c}$$, then we’d find a vector $$\vec{C}$$ orthogonal to both $$\vec{A}$$ and $$\vec{B}$$ by subtracting from $$\vec{c}$$ its components in the $$\vec{A}$$ and $$\vec{B}$$ directions. Therefore

$$\vec{C}=\vec{c}-\dfrac{\vec{A}^T\vec{c}}{\vec{A}^T\vec{A}}\vec{A}-\dfrac{\vec{B}^T\vec{c}}{\vec{B}^T\vec{B}}\vec{B} \qquad \text{ and}\qquad \vec{q_3}=\dfrac{\vec{C}}{\\|\vec{C}\\|}$$

For the vectors $$\vec{a}$$, $$\vec{b}$$ and $$\vec{c}$$ we get the orthonormal matrix

$$ Q=\begin{bmatrix} \vec{q_1} & \vec{q_2} & \vec{q_3} \end{bmatrix}$$

The columnspace of $$Q$$ is the plane spanned by the vectors $$\vec{a}$$, $$\vec{b}$$ and $$\vec{c}$$.
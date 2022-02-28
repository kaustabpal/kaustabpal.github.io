---
layout: post
title: "System of Linear Equation"
permalink: "system_linear_equation"
date: 2022-02-28 19:05
tags: [""]
categories:
description: ""
---

The general form of a system of linear equations is:

$$ a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n = b_1\\ a_{21}x_1 + a_{22}x_2 +
\dots + a_{2n}x_n = b_2\\ \vdots\\ a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n =
b_m\\ $$

The above system can be written in terms of vectors as:

$$ \begin{bmatrix}a_{11}\\a_{21}\\ \vdots \\ a_{m1}\end{bmatrix}x_1 +
\begin{bmatrix}a_{11}\\a_{22}\\ \vdots \\ a_{m2}\end{bmatrix}x_2 + \dots +
\begin{bmatrix}a_{1n}\\a_{2n}\\ \vdots \\ a_{mn}\end{bmatrix}x_n =
\begin{bmatrix}b_{1}\\b_{2}\\ \vdots \\ b_{m}\end{bmatrix} $$

This in turn can be written as:

$$ \underbrace{\begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{11} &
a_{12} & \dots & a_{1n} \\ \vdots & \vdots & \dots & \vdots \\ a_{11} & a_{12} &
\dots & a_{1n} \end{bmatrix}}_\text{matrix A} \underbrace{\begin{bmatrix}x_1\\
x_2\\ \vdots \\ x_n \end{bmatrix}}_\text{vector \textbf{x}} =
\underbrace{\begin{bmatrix}b_1\\ b_2\\ \vdots \\ b_m \end{bmatrix}}_\text{vector
\textbf{b}} $$

Solving a system of linear equation means finding the vector **x** that
satisfies the equation $$A\textbf{x}=\textbf{b}$$. 

Multiplying a matrix $$A$$ by a vector $$\boldsymbol{x}$$ gives us a vector
which is a [ [Linear Combination] ]( {% post_url 2022-02-28-linear_combination
%} ) of the column vectors of the matrix $$A$$ where the co-efficients of the
linear combination are the elements of the vector $$\boldsymbol{x}$$. Therefore
for $$A\boldsymbol{x}=\boldsymbol{b}$$ to have a solution $$\boldsymbol{x}$$,
$$\boldsymbol{b}$$ must be a linear combination of the column vectors of $$A$$
where the co-efficients of the linear combination are the elements of the
solution vector $$\boldsymbol{x}$$. In other words, $$\boldsymbol{b}$$ must lie
in the [ [Column Space] ]( {% post_url 2022-02-28-column_space %} ) of $$A$$.

Geometrically, finding the solution for the system of equations means finding
the points where the the lines/planes intersect.

- If the lines/planes intersect at only one point, then the system of linear
  equations will have only one solution.

- If the lines/planes doesnot intersect at any point, then the system of
  equations will not have any solution. Ex: Parallel lines/planes.

- If the lines/planes lie on top of each other, they intersect at infinite
  points and therefore the system of linear equations will have infinite
  solutions.

### Elimination with Matrices

To solve a system of linear equations we need to solve for the unknowns. Elimination is the technique most commonly used by computer softwares to solve systems of linear equations. It finds a solution $$\boldsymbol{x}$$ to $$A\boldsymbol{x}=\boldsymbol{b}$$ if $$A$$ has no dependent rows or columns.

For explaining the elimination technique, let us take an example. Suppose we have a system of equations as

$$x+2y+z=2$$

$$3x+8y+z=12$$

$$4y+z=2$$

In this example, we have

$$A=\begin{bmatrix}1 & 2 & 1\\ 3 & 8 & 1\\ 0 & 4 & 1\end{bmatrix}$$

$$\boldsymbol{b}=\begin{bmatrix}2\\ 12\\ 2\end{bmatrix}$$

$$\boldsymbol{x}=\begin{bmatrix}x\\ y\\ z\end{bmatrix}$$

Our goal is to perform operations on matrix $$A$$ and transform the matrix into an echelon form $$U$$. The operations that will be done on matrix $$A$$ will also be done on the vector $$\boldsymbol{b}$$. The reason we want to transform matrix $$A$$ to $$U$$ is because it becomes easier to solve for the unknowns using back substitution.

The number $$1$$ in the upper left corner of matrix $$A$$ is called the first pivot. We multiply the first row with an appropriate value which in this case is $$3$$ and subtract it from the second row. The first number in the second row now becomes $$0$$ and the matrix becomes

$$\begin{bmatrix}1 & 2 & 1\\ 0 & -2 & 2\\ 0 & 4 & 1\end{bmatrix}$$

The number $$-2$$ in the second row now becomes the second pivot. We multiply the second row by the appropriate value which in this case is $$2$$ and add it to the third row. The first number in the third row was already $$0$$ but after this operation the second number in the third row also becomes $$0$$ and we achieve an upper triangular matrix. The number $$5$$ in the third row is called the third pivot.

$$\begin{bmatrix}1 & 2 & 1\\ 0 & -2 & 2\\ 0 & 0 & 5\end{bmatrix}$$

The whole purpose of elimination is to go from $$A$$ to $$U$$. The operations that we performed here to transform $$A$$ to $$U$$ is only specific to our current matrix $$A$$ and it will vary depending on the values of the matrix.

While we did all these operations on $$A$$, we also need to do those same operations on $$\boldsymbol{b}$$. Thus after performing the operations on $$\boldsymbol{b}$$ we get the new vector $$\boldsymbol{c}$$ as

$$\boldsymbol{c}=\begin{bmatrix}2\\ 6\\ -10\end{bmatrix}$$

After elimination, our new system of equations becomes:

$$\begin{bmatrix}1 & 2 & 1\\ 0 & -2 & 2\\ 0 & 0 & 5\end{bmatrix}\begin{bmatrix}x\\ y\\ z\end{bmatrix}=\begin{bmatrix}2\\ 6\\ -10\end{bmatrix}$$

which can also be written as

$$x+2y+z=2$$

$$-2y+2z=6$$

$$5z=-10$$

Performing back substitution on these equations, we can easily get the solution of the system as

$$\begin{bmatrix}x\\ y\\ z\end{bmatrix}=\begin{bmatrix}2 \\ 1 \\ -2 \end{bmatrix}$$

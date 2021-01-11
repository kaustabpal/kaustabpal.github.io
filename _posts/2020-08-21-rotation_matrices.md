---
layout: post
title: "Rotation Matrices"
excerpt: "An explanation about how a vector's coordinate system is rotated with respect to a reference coordinate system."
tags: maths
---
- *For the sake of making the explanations easier, we will be working on the two dimensional coordinate plane $$R^2$$. The same explanations work on higher-dimensional spaces as well.*

- *While the concept of rotation matrix is very simple, it might get a bit confusing if you are learning about it for the first time. My advice would be to follow along slowly with this or any other source you are using and do the math yourself on a piece of paper.*

{: class="table-of-content"}
* TOC
{:toc}

# Vectors

Vectors are representations of points in a space. The points in the 2-d space are represented by their $$x$$ and $$y$$ coordinates which can be written in the form of a vector as: 

$$\begin{bmatrix}x \\ y \end{bmatrix}$$

Vectors are represented as lower-case **bold** letters.

# Vector dot product

Given two vectors $$\boldsymbol{a}$$ and $$\boldsymbol{b}$$, their dot products is given as $$\rVert \boldsymbol{a} \rVert \times \rVert \boldsymbol{b} \rVert \times \cos \theta$$. Geometrically, the dot product tells us the projection of vector $$\boldsymbol{a}$$ on the vector $$\boldsymbol{b}$$.

# Matrices 

A matrix can be seen as a set of column vectors where the number of columns in the matrix is the number of column vectors and the number of rows in the matrix is the size of each column vector. A $$m\times n$$ matrix has $$n$$ column vectors and each column vector is of length $$m$$. Matrices are represented as UPPERCASE letters.

# Linear combination

Given two vectors $$\boldsymbol{v_{1}}$$ and $$\boldsymbol{v_2}$$ and scalars $$\lambda_1$$ and $$\lambda_2$$, the sum $$\lambda_1 \boldsymbol{v_1}+\lambda_2 \boldsymbol{v_2}$$ is called a linear combination of the vectors $$\boldsymbol{v_1}$$ and $$\boldsymbol{v_2}$$. The scalars $$\lambda_1$$ and $$\lambda_2$$ are called the coefficients of the linear combination. 

If vectors $$\boldsymbol{v_1}$$ and $$\boldsymbol{v_2}$$ are column vectors of a matrix $$A$$ and the vector $$\boldsymbol{x}$$ is given by $$\boldsymbol{x}=\begin{bmatrix}\lambda_1 & \lambda_2\end{bmatrix}^T$$, then $$A\boldsymbol{x}$$ gives us the linear combination of the column vectors of the matrix $$A$$ where the coefficients of the linear combination are the values of the vector $$\boldsymbol{x}$$.

# Vector spaces

In two dimensional vectors, the vectors have two components that represents the $$x$$ coordinates and $$y$$ coordinates of the two-dimensional plane. This two-dimensional plane is called a vector space. If we pick any two vectors from this two-dimensional plane, their linear combination will always lie in the plane. As a rule of thumb, if we take any number of vectors from a vector space, their linear combination will always lie in that vector space. Some other examples of vector spaces are the three-dimensional space, four-dimensional space and so on.

# Span

We are given two vectors $$\begin{bmatrix}0 & 1\end{bmatrix}^T$$ and $$\begin{bmatrix}1 & 0\end{bmatrix}^T$$. These two vectors lie on the two-dimensional vector space. If we pick any vector from this space, it will be a linear combination of these two vectors. That is why these two vectors are said to span the space. 
 
 Vectors $$\boldsymbol{v_1}, \boldsymbol{v_2}, \dots \boldsymbol{v_n}$$ are said to span a space when the space consists of all linear combinations of those vectors. 
# Basis

 If we are given a sequence of vectors $$\boldsymbol{v_1}, \boldsymbol{v_2}, \dots \boldsymbol{v_n}$$ such that they are independent and they span a vector space, then these vectors are said to form the basis for that vector space. 

# Standard Basis

The standard basis vectors are the simplest basis vectors for a given vector space. In a standard basis vector, only one element is $$1$$ and all the other elements are $$0$$.

For a given euclidean space, the standard basis vectors are the unit vectors of the axes of that vector space. For the two dimensional vector space, the standard basis vectors are:

$$\hat{x}=\begin{bmatrix}1\\0\end{bmatrix}$$

and 

$$\hat{y}=\begin{bmatrix}0\\1\end{bmatrix}$$

Every other vector in this vector space is given by the linear combination of these two vectors.

# Coordinate system

We represent a coordinate system as $$\{A\}$$.

Let $$\{E\}$$ be the coordinate system of the earth and $$\{P\}$$ be the coordinate system of a flying airplane.

Let $$^{P}\boldsymbol{p}$$ be the position vectors of the people sitting inside the plane in the airplane's coordinate system $$\{P\}$$.

Let $$^{E}\boldsymbol{p}$$ be the position vector of the people sitting inside the plane in the earth's coordinate system $$\{E\}$$.

As the airplane moves, $$^{P}\boldsymbol{p}$$ remains the same, but $$^{E}\boldsymbol{p}$$ changes. This is because when we say the airplane moves, what we are implying is the coordinate system of the airplane i.e. $$\{P\}$$ is moving with respect to the coordinate system of the earth $$\{E\}$$. What this means is that as the airplane moves, everything inside the airplane moves with it as well since they lie in the airplane's coordinate system. That is why with respect to the airplane's coordinate system everything is stationary. However with respect to the earth's coordinate system, as the airplane moves, everything inside the airplane is moving as well. Therefore their position vectors with respect to the earth's coordinate system will keep changing.

# Rotating a coordinate frame

Let's start by looking at the airplane's coordinate system $$\{P\}$$, which is at an angle $$0^{\circ}$$ with respect to the earth's coordinate system $$\{E\}$$.

<figure>
<img src="assets/img/mr/File_001.png" style="width:70%;" />
</figure>

{:.caption}
Fig 1: Airplane coordinate system $$\{P\}$$ at an angle $$0^{\circ}$$ with respect to the Earth coordinate system $$\{E\}$$


For the airplane's coordinate system, $$^{P}\boldsymbol{x}$$ is the unit basis vector which represents the x-axis and $$^{P}\boldsymbol{y}$$ is the unit basis vector which represents the y-axis. 

For the earth's coordinate system, $$^{E}\boldsymbol{x}$$ is the unit basis vector which represents the x-axis and $$^{E}\boldsymbol{y}$$ is the unit basis vector which represents the y-axis.

Since $$^{P}\boldsymbol{x}$$ is the standard basis vector for $$\{P\}$$,

$$^{P}\boldsymbol{x} = \begin{bmatrix}1\\0\end{bmatrix}$$

To represent $$^{P}\boldsymbol{x}$$ with respect to earth's coordinate system $$\{E\}$$, we need to find the projection of $$^{P}\boldsymbol{x}$$ on $$^{E}\boldsymbol{x}$$ and $$^{P}\boldsymbol{x}$$ on $$^{E}\boldsymbol{y}$$. This will give us the $$x$$ and $$y$$ coordinates of $$^{P}\boldsymbol{x}$$ with respect to earth's coordinate system $$\{E\}$$.

To find the projection of one vector on another we use dot product. 

Therefore the $$x$$ component of $$^{P}\boldsymbol{x}$$ with respect to $$\{E\}$$ is 

$$\rVert ^{P}\boldsymbol{x}\rVert \rVert ^{E}\boldsymbol{x}\rVert \cos \theta$$

The $$y$$ component of $$^{P}\boldsymbol{x}$$ with respect to $$\{E\}$$ is 

$$\rVert ^{P}\boldsymbol{x}\rVert \rVert ^{E}\boldsymbol{y}\rVert \cos (90 - \theta)$$

Representing this in the form of a vector, we get:

$$^{E}\boldsymbol{x}_{P}=\begin{bmatrix}\rVert ^{P}\boldsymbol{x}\rVert \rVert ^{E}\boldsymbol{x}\rVert \cos \theta \\ \rVert ^{P}\boldsymbol{x}\rVert \rVert ^{E}\boldsymbol{y}\rVert \cos (90 - \theta)\end{bmatrix}$$

$$^{E}\boldsymbol{x}_{P}$$ is the $$^{P}\boldsymbol{x}$$ vector with respect to Earth's coordinate system.

Here, $$\theta$$ is the angle between $$^{P}\boldsymbol{x}$$ and $$^{E}\boldsymbol{x}$$. For this example $$\theta$$ is $$0$$. 

Since $$^{P}\boldsymbol{x}$$ and $$^{E}\boldsymbol{x}$$ are unit basis vectors, $$\rVert ^{P}\boldsymbol{x}\rVert = 1$$ and $$\rVert ^{E}\boldsymbol{x}\rVert = 1$$.

Putting these values in $$^{E}\boldsymbol{x}_{P}$$ we get

$$^{E}\boldsymbol{x}_{P}=\begin{bmatrix}1 \times 1 \times \cos 0 \\ 1 \times 1 \times \cos (90 - 0)\end{bmatrix} = \begin{bmatrix}1 \\ 0\end{bmatrix}$$ 

Similarly, since $$^{P}\boldsymbol{y}$$ is the standard basis vector for $$\{P\}$$,

$$^{P}\boldsymbol{y} = \begin{bmatrix}0\\1\end{bmatrix}$$

To represent $$^{P}\boldsymbol{y}$$ with respect to earth's coordinate system, we need to find the projection of $$^{P}\boldsymbol{y}$$ on $$^{E}\boldsymbol{x}$$ and $$^{P}\boldsymbol{y}$$ on $$^{E}\boldsymbol{y}$$. This will give us the $$x$$ and $$y$$ coordinates of $$^{P}\boldsymbol{y}$$ with respect to earth's coordinate system $$\{E\}$$.

The $$x$$ component of $$^{P}\boldsymbol{y}$$ with respect to $$\{E\}$$ is 

$$\rVert ^{P}\boldsymbol{y}\rVert \rVert ^{E}\boldsymbol{x}\rVert \cos (90+\theta)$$

The $$y$$ component of $$^{P}\boldsymbol{y}$$ with respect to $$\{E\}$$ is 

$$\rVert ^{P}\boldsymbol{y}\rVert \rVert ^{E}\boldsymbol{y}\rVert \cos \theta$$

Representing this in the form of a vector, we get:

$$^{E}\boldsymbol{y}_{P}=\begin{bmatrix}\rVert ^{P}\boldsymbol{y}\rVert \rVert ^{E}\boldsymbol{x}\rVert \cos (90+\theta) \\ \rVert ^{P}\boldsymbol{y}\rVert \rVert ^{E}\boldsymbol{y}\rVert \cos \theta\end{bmatrix}$$

$$^{E}\boldsymbol{y}_{P}$$ is the $$^{P}\boldsymbol{y}$$ vector with respect to Earth's coordinate system $$\{E\}$$.

Here, $$\theta$$ is the angle between $$^{P}\boldsymbol{x}$$ and $$^{E}\boldsymbol{x}$$. Since $$^{P}\boldsymbol{x}$$ makes an angle of $$\theta$$ with $$^{E}\boldsymbol{x}$$, $$^{P}\boldsymbol{y}$$ will make an angle of $$90+\theta$$ with $$^{E}\boldsymbol{x}$$ and $$^{P}\boldsymbol{y}$$ will make an angle of $$\theta$$ with $$^{E}\boldsymbol{y}$$. For this example $$\theta$$ is $$0$$. 

Since $$^{P}\boldsymbol{y}$$, $$^{E}\boldsymbol{x}$$ and $$^{E}\boldsymbol{y}$$ are unit basis vectors, $$\rVert ^{P}\boldsymbol{x}\rVert = 1$$ and $$\rVert ^{E}\boldsymbol{x}\rVert = 1$$ and $$\rVert ^{E}\boldsymbol{y}\rVert = 1$$.

Putting these values in $$^{E}\boldsymbol{y}_{P}$$ we get

$$^{E}\boldsymbol{y}_{P}=\begin{bmatrix}1 \times 1 \times \cos (90+0) \\ 1 \times 1 \times \cos 0\end{bmatrix} = \begin{bmatrix}0 \\ 1\end{bmatrix}$$

We represent the coordinate system $$\{P\}$$ rotated by an angle $$0^{\circ}$$ with respect to coordinate system $$\{E\}$$ in the form of a matrix as:

$$^{E}R_{P} = \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix}$$

Here $$^{E}R_{P}$$ is called a rotation matrix. To represent any vector in $$\{P\}$$ with respect to $$\{E\}$$ we simply multiply that vector with this matrix. 

Suppose we have a vector in $$\{P\}$$ represented as 

$$^{P}\boldsymbol{v} = \begin{bmatrix}x\\y\end{bmatrix}$$

To represent $$^{P}\boldsymbol{v}$$ with respect to $$\{E\}$$ we multiply $$^{P}\boldsymbol{v}$$ to our rotation matrix $$^{E}R_{P}$$

$$^{E}\boldsymbol{v}_{P}= ^{E}R_{P} \times ^{P}\boldsymbol{v} $$

$$\implies ^{E}\boldsymbol{v}_{P}= \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix} \times \begin{bmatrix}x\\y\end{bmatrix}$$

$$ \implies ^{E}\boldsymbol{v}_{P}= x \times \begin{bmatrix}1\\0\end{bmatrix} + y \times \begin{bmatrix}0 \\ 1\end{bmatrix}$$

$$ \implies ^{E}\boldsymbol{v}_{P}= x \times ^{E}\boldsymbol{x}_{P} + y \times ^{E}\boldsymbol{y}_{P}$$

What happens here is we scale the standard basis vectors of our rotated coordinate frame $$\{P\}$$ with the $$x$$ and $$y$$ values of the vector inside that coordinate frame $$\{P\}$$. Adding these will give us the vector $$^{E}\boldsymbol{v}_{P}$$ which is $$^{P}\boldsymbol{v}$$ rotated by angle $$\theta$$ with respect to $$\{E\}$$. This also means that the new rotated vector is just the linear combination of the standard basis vectors of our rotated coordinate system $$\{P\}$$ which means **we cannot rotate a vector without rotating it's coordinate system with respect to our reference coordinate system.** In our example our reference coordinate system is the earth coordinate system $$\{E\}$$. 

In our example, we rotate our coordinate system $$\{P\}$$ by an angle $$0 ^{\circ}$$ with respect to $$\{E\}$$. The same technique applies for rotating $$\{P\}$$ by any angle $$\theta$$. To make things easier, we can simply formulate a generalized rotation matrix as 

$$^{E}R_{P} = \begin{bmatrix}\rVert ^{P}\boldsymbol{x}\rVert \rVert ^{E}\boldsymbol{x}\rVert \cos \theta & \rVert ^{P}\boldsymbol{y}\rVert \rVert ^{E}\boldsymbol{x}\rVert \cos (90+\theta) \\ \rVert ^{P}\boldsymbol{x}\rVert \rVert ^{E}\boldsymbol{y}\rVert \cos (90 - \theta) & \rVert ^{P}\boldsymbol{y}\rVert \rVert ^{E}\boldsymbol{y}\rVert \cos \theta \end{bmatrix}$$

Now $$^{P}\boldsymbol{x}$$, $$^{P}\boldsymbol{y}$$ and $$^{E}\boldsymbol{x}$$, $$^{E}\boldsymbol{y}$$ are all standard basis vectors. Therefore $$\rVert ^{P}\boldsymbol{x}\rVert$$, $$\rVert ^{P}\boldsymbol{y}\rVert$$ and $$\rVert ^{E}\boldsymbol{x}\rVert$$, $$\rVert ^{E}\boldsymbol{y}\rVert$$ are all $$1$$.

Therefore we can write $$^{E}R_{P}$$ as

$$^{E}R_{P} = \begin{bmatrix}\cos \theta & \cos (90+\theta) \\ \cos (90 - \theta) & \cos \theta \end{bmatrix}$$

From trigonometry, we know $$\cos (90+\theta) = - \sin \theta$$ and $$\cos (90-\theta) =  \sin \theta$$. Substituting these in $$^{E}R_{P}$$ we get our final rotation matrix as:

$$^{E}R_{P} = \begin{bmatrix}\cos \theta & - \sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}$$

# Rotation Matrices

Using the rotation matrix $$^{E}R_{P}$$ we can rotate any vector's coordinate system by any angle with respect to our reference coordinate system. 

In our airplane example, our reference coordinate system is the earth $$\{E\}$$. If we want to rotate our airplane by an angle of $$\theta$$ with respect to Earth $$\{E\}$$, we need to rotate our airplane's coordinate system $$\{P\}$$ by an angle of $$\theta$$ with respect to our Earth's coordinate system $$\{E\}$$.

Let $$\theta$$ be equal to $$30 ^{\circ}$$.

Then our rotation matrix $$^{E}R_{P}$$ becomes

$$^{E}R_{P} = \begin{bmatrix}\cos 30 ^{\circ} & - \sin 30 ^{\circ} \\ \sin 30 ^{\circ} & \cos 30 ^{\circ} \end{bmatrix}$$

$$\implies ^{E}R_{P} = \begin{bmatrix}0.86 & - 0.5 \\ 0.5 & 0.86 \end{bmatrix}$$

If we plot the column vectors of this rotation matrix, we will see that the coordinate system of the airplane $$\{P\}$$ is rotated by $$30^{\circ}$$ with respect to the Earth's coordinate system $$\{E\}$$. Any vector that lies in $$\{P\}$$ will also be rotated by $$30 ^{\circ}$$ with respect to $$\{E\}$$. To represent any vector in $$\{P\}$$ with respect to $$\{E\}$$, we simply multiply that vector with our rotation matrix $$^{E}R_{P}$$ to get the vector with respect to our earth coordinate system $$\{E\}$$.
<figure>
<img src="assets/img/mr/File_002.png" style="width:70%;" />
</figure>

{:.caption}
Fig 2: Airplane coordinate system $$\{P\}$$ at an angle $$30^{\circ}$$ with respect to the Earth coordinate system $$\{E\}$$.


# References

1. [https://www.youtube.com/watch?v=lVjFhNv2N8o](https://www.youtube.com/watch?v=lVjFhNv2N8o)

2. Craig, 1955. (1986). Introduction to robotics : mechanics & control / John J. Craig.. Reading, Mass.: Addison-Wesley Pub. Co.,. ISBN: 0201103265


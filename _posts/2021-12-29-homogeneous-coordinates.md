--- 
layout: post 
title: "Homogeneous Coordinates" 
permalink: "homogeneous_coordinates"
tags: [""] 
categories: notes
---

{:class="table-of-content"}
* TOC 
{:toc}

## Definition

The representation of a point $$X$$ of a geometric object is in the homogeneous coordinates if
$$X$$ and $$\lambda X$$ represent the same object for $$\lambda \neq 0$$. 

## Explanation

Homogeneous coordinate is a coordinate system used in projective geometry. Any point in the projective plane is represented by homogeneous coordinates. The euclidean plane is a subset of the projective plane and is at a distance of $$1$$ from the origin of the projective plane. Then the origin of euclidean coordinate in the projective plane can be written in the homogeneous coordinate as $$\begin{bmatrix}0 & 0 & 1 \end{bmatrix}$$. Any point $$\begin{bmatrix}x & y\end{bmatrix}$$ in the euclidean plane can be represented in the projective plane
by simply adding a third coordinate $$1$$ at the end. The point then becomes $$\begin{bmatrix}x & y & 1\end{bmatrix}$$.

If the homogeneous coordinates are multiplied by a scalar, they will still represent the same coordinates in the euclidean space. 

Example:

$$\begin{bmatrix}x \\ y \\ 1 \end{bmatrix} = w*\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix}wx \\ wy \\ w\end{bmatrix} = \begin{bmatrix}u \\ v \\ w\end{bmatrix}$$


To get the euclidean coordinates from the 3d homogeneous coordinates we divide
the homogeneous coordinates by $$w$$ ie $$\begin{bmatrix}u/w & v/w & 1\end{bmatrix}$$. Then the euclidean
coordinates will become $$\begin{bmatrix}u/w & v/w \end{bmatrix}$$.

## Advantages

An advantage of using homogeneous coordinates is that we can easily use multiple
transformations by concatinating several matrix vector multiplications. 

In the euclidean space, given a point we can scale it and rotate it and
translate it but we can't do all these operations at the same time as scaling
and rotation requires a matrix multiplication whereas translation requires a
vector addition. To combine all these operations into a single matrix we
transform the euclidean coordinate to homogeneous coordinate and now the
translation operation also becomes a matrix multiplication as 

$$ \begin{bmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1 \end{bmatrix}\begin{bmatrix} x \\ y \\ 1\end{bmatrix}$$

Thus we can now have one single transformation matrix by combining all the
transformations together. We can save a lot of computational cost this way.

It also allows us to represent points that are infinitely far away using finite
numbers. For example to represent a point in infinity we have to write it as
$$\begin{bmatrix}x & y & 0\end{bmatrix}$$.

## Lines in images (2d)

Given a point in 3d homogeneous coordinates, we can check if it lies on  a line
in the 2d image. We can also find the 3d point of intersection of two lines in 
an image. And given two 3d points, we can find the line in the image joining those points. 

A line is represented in the standard form as $$ax + by + c = 0$$.

We can write this in the vector form as $$ [a, b, c]^T [x, y, 1]$$ Therefore if
we know the coefficients of a line then we can multiply the $$u/w$$ and $$v/w$$ of a
point in homogeneous coordinates and if we get 0 then that point in 3d lies on that
line in that image.

### How to find intersection of two lines?

If we know the coefficients of two lines $$l$$ and $$m$$ and we want to find the
point of intersection of the two lines, then that point $$X$$ will be such that
$$l^TX=0$$ and $$m^TX=0$$. Therefore we can write a matrix as 
$$ \begin{bmatrix} l1 & l2 & l3 \\ 
m1 & m2 & m3 \end{bmatrix}\begin{bmatrix}x \\ y \\ 1 \end{bmatrix} = 0$$

### How to find the line joining two points?
We know $$X$$ and $$Y$$ of two points in the homogeneous coordinates. We have to
find the $$L$$. Then we can write this as $$\begin{bmatrix}x_1 & x_2 & 1\\
y_1 & y_2 & 1\end{bmatrix} \begin{bmatrix}l_1\\ l_2 \\ l_3\end{bmatrix}= 
\begin{bmatrix}0\\ 0\end{bmatrix}$$

### Lines passing through a point at infinity

In homogeneous coordinates a point in infinity is represented as $$[x, y , 0]$$.
If we want to find all the lines that pass through this point we need to do
$$\begin{bmatrix} x & y & 0 \end{bmatrix} \begin{bmatrix}l1 \\ l2 \\
l3\end{bmatrix} = \begin{bmatrix}0\\0\\0\end{bmatrix}$$ Here we will get a fixed
$$l1$$ and $$l2$$ but the $$l3$$ can be anything thus proving that all the
parallel lines meet at a point in infinity.

### Ideal line

The ideal line represented as $$[0, 0, 1]$$ contains all the points at infinity.
This line is also called the horizon. Thus any points at infinity $$[x, y, 0]$$
lies on this line as $$\begin{bmatrix}x & y & 0 \end{bmatrix}\begin{bmatrix}0\\
0\\ 1\end{bmatrix} = \begin{bmatrix}0 \\ 0 \\ 0 \end{bmatrix}$$

The same concepts can be represented in 3d as well where the homogeneous
coordinates will be four dimensional. In $$3d$$ we will have planes instead of a
line and sky instead of horizon.

## Resources

* [An interactive guide to homogeneous coordinates](https://wordsandbuttons.online/interactive_guide_to_homogeneous_coordinates.html)
* [Homogeneous Coordinates (Cyrill Stachniss, 2020)](https://www.youtube.com/watch?v=MQdm0Z_gNcw&t=572s)


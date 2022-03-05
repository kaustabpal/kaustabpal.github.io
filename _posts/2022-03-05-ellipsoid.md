---
layout: post
title: "Ellipsoid"
permalink: "ellipsoid"
date: 2022-03-05 21:12
tags: [""]
categories:
description: ""
---

{:class="table-of-content"}
* TOC 
{:toc}

These belong to the family of [ [Euclidean Balls] ]( {% post_url
2022-03-03-euclidean_ball %} ). In $$R^2$$, the ellipsoid is represented by the
standard equation $$\dfrac{(x - x_c)^2}{a^2} + \dfrac{(y-y_c)^2}{b^2} \leq r$$.
Here $$x_c$$ and $$y_c$$ are the $$x$$ and the $$y$$ coordinates of the center
of the ellipse, $$a$$ is the semi-major axis and $$b$$ is the semi-minor axis.
We can represent the standard form equation in the matrix form as 

$$\begin{bmatrix} x - x_c & y-y_c \end{bmatrix}^T \begin{bmatrix}a & 0 \\ 0 & b
\end{bmatrix}^{-1}\begin{bmatrix}x-x_c \\ y- y_c\end{bmatrix} \preceq r$$ 

The matrix $$\begin{bmatrix}a & 0 \\ 0 & b \end{bmatrix}$$ referred to as $$P$$
is a [ [Symmetric Matrix] ]( {% post_url 2022-03-05-symmetric_matrix %} ) and
also a [ [Positive Definite Matrix] ]( {% post_url
2022-03-05-positive_definite_matrix %} ). It determines how far the ellipsoid
extends in each direction from the center of the ellipse. The length of the
semi-axis of the ellipse is given by the $$\sqrt \lambda _i$$ where
$$\lambda_i$$s are the [ [Eigen Value]s ]( {% post_url 2022-03-05-eigen_value %}
) of the matrix $$P$$.

An ellipsoids is a [ [Convex Set] ]( {% post_url 2022-02-19-convex_set %} )

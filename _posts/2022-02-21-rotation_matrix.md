---
layout: post
title: "Rotation Matrix"
permalink: "rotation_matrix"
date: 2022-02-21 14:35
moddate: 2022-02-21 14:35
tags: [""]
categories:
---

Given a coordinate frame $$\{B\}$$ which is at an angle of $$\theta$$ with
respect to $$\{A\}$$ and there is no translation between $$\{B\}$$ and
$$\{A\}$$, we can represent the points in $$\{B\}$$ with respect to $$\{A\}$$ by
multiplying them with the rotation matrix $$^{A}_{B}{R}$$. Here $$^{A}_{B}{R}$$
is a $$3\times 3$$ matrix that maps a vector from coordinate frame $$B$$ to
coordinate frame $$A$$.

We can get the rotation matrix by projecting the unit vectors of the principle
axes of $$\{B\}$$ to the unit vectors of the principle axes of $$\{A\}$$ and
stacking them horizontally in the form of a matrix as
$$\begin{bmatrix}^{A}\hat{X}_{B}, ^{A}\hat{Y}_{B},
^{A}\hat{Z}_{B}\end{bmatrix}$$.

We can expand $$\begin{bmatrix}^{A}\hat{X}_{B}, ^{A}\hat{Y}_{B},
^{A}\hat{Z}_{B}\end{bmatrix}$$ as 

$$ \begin{align*}

^{A}_{B}{R} &= \begin{bmatrix}^{A}\hat{X}_{B}, ^{A}\hat{Y}_{B},
^{A}\hat{Z}_{B}\end{bmatrix} \\ &= \begin{bmatrix} ^{B}\hat{X}^{A}\hat{X} &&
^{B}\hat{Y}^{A}\hat{X} &&^{B}\hat{Z}^{A}\hat{X} \\ ^{B}\hat{X}^{A}\hat{Y} &&
^{B}\hat{Y}^{A}\hat{Y} && ^{B}\hat{Z}^{A}\hat{Y} \\ ^{B}\hat{X}^{A}\hat{Z} &&
^{B}\hat{Y}^{A}\hat{Z} && ^{B}\hat{Z}^{A}\hat{Z} \end{bmatrix} \end{align*} $$

If we are rotating along the $$z$$ axis, then only the $$^{B}\hat{X}$$ and
$$^{B}\hat{Y}$$ axes rotate by an angle $$\theta$$ where as the angle between
$$^{B}\hat{Z}$$ and $$^{A}\hat{Z}$$ will be $$0$$. In this case we represent the
rotation matrix as $$^{A}_{B}{R}_z$$ as 

$$ \begin{align*} ^{A}_{B}{R}_z &= \begin{bmatrix}^{A}\hat{X}_{B},
^{A}\hat{Y}_{B}, ^{A}\hat{Z}_{B}\end{bmatrix} \\

&= \begin{bmatrix} ^{B}\hat{X}^{A}\hat{X} && ^{B}\hat{Y}^{A}\hat{X}
&&^{B}\hat{Z}^{A}\hat{X} \\ ^{B}\hat{X}^{A}\hat{Y} && ^{B}\hat{Y}^{A}\hat{Y} &&
^{B}\hat{Z}^{A}\hat{Y} \\ ^{B}\hat{X}^{A}\hat{Z} && ^{B}\hat{Y}^{A}\hat{Z} &&
^{B}\hat{Z}^{A}\hat{Z} \end{bmatrix} \\

&= \begin{bmatrix} \mid ^{B}\hat{X}\mid \mid^{A}\hat{X}\mid \cos \theta && \mid
^{B}\hat{Y}\mid \mid^{A}\hat{X}\mid \cos (90 + \theta) && \mid^{B}\hat{Z}\mid
\mid^{A}\hat{X}\mid \cos 90 \\ \mid ^{B}\hat{X}\mid \mid^{A}\hat{Y}\mid \cos
(90- \theta) && \mid ^{B}\hat{Y}\mid \mid ^{A}\hat{Y}\mid \cos \theta && \mid
^{B}\hat{Z}\mid \mid ^{A}\hat{Y}\mid \cos 90 \\ \mid ^{B}\hat{X}\mid
\mid^{A}\hat{Z}\mid \cos 90 && \mid ^{B}\hat{Y}\mid \mid^{A}\hat{Z}\mid \cos 90
&& \mid ^{B}\hat{Z}\mid \mid ^{A}\hat{Z}\mid \cos 0 \end{bmatrix} \end{align*}
$$

Now we know that $$^{B}\hat{X}$$, $$^{B}\hat{Y}$$, $$^{B}\hat{Z}$$, and
$$^{A}\hat{X}$$, $$^{A}\hat{Y}$$, $$^{A}\hat{Z}$$ are unit basis vectors.
Therefore $$\mid ^{B}\hat{X}\mid$$, $$\mid ^{B}\hat{Y}\mid$$, $$\mid
^{B}\hat{Z}\mid$$ and $$\mid ^{A}\hat{X}\mid$$, $$\mid ^{A}\hat{Y}\mid$$, $$\mid
^{A}\hat{Z}\mid$$ will be equal to $$1$$.

Therefore we can write

$$ \begin{align*} ^{A}_{B}{R}_z &= \begin{bmatrix} \cos \theta && -\sin \theta
&& 0\\ \sin \theta && \cos \theta && 0\\ 0 && 0 && 1 \end{bmatrix} \end{align*}
$$

Similarly we find  $$ ^{A}_{B}R_y = \begin{bmatrix} \cos \theta && 0 && \sin
\theta \\ 0 && 1 && 0 \\ -sin \theta && 0 && \cos \theta \end{bmatrix}$$ if we
are rotating along the $$y$$ axes.

If we are rotating along the $$x$$ axes, $$^{A}_{B}R_x = \begin{bmatrix} 1 && 0
&& 0 \\ 0 && \cos \theta && -\sin \theta \\ 0 && sin \theta && \cos \theta
\end{bmatrix}$$.


Rotation matrices are invertible and the inverse of a rotation matrix $$R$$ is
$$R^T$$. 

$$ \begin{align*}

^{B}_{A}{R} &= \begin{bmatrix}^{B}\hat{X}_{A}, ^{B}\hat{Y}_{A},
^{B}\hat{Z}_{A}\end{bmatrix} \\ &= \begin{bmatrix} ^{A}\hat{X}^{B}\hat{X} &&
^{A}\hat{Y}^{B}\hat{X} && ^{A}\hat{Z}^{B}\hat{X} \\ ^{A}\hat{X}^{B}\hat{Y} &&
^{A}\hat{Y}^{B}\hat{Y} && ^{A}\hat{Z}^{B}\hat{Y} \\ ^{A}\hat{X}^{B}\hat{Z} &&
^{A}\hat{Y}^{B}\hat{Z} && ^{A}\hat{Z}^{B}\hat{Z} \end{bmatrix}\\ &=
^{A}_{B}{R}^T \end{align*} $$

Thus if we use $$R$$ to represent points in $$B$$ in terms of $$A$$ then we can
use $$R^{T}$$ to represent points in $$A$$ in terms of $$B$$.


They are also used to describe the [ [Orientation] ]( {% post_url
2022-02-21-orientation %} ) of a point in space. 

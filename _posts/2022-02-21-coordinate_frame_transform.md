---
layout: post
title: "Coordinate Frame Transform"
permalink: "coordinate_transform"
idate: 2022-02-23 00:05
date: 2022-02-23 00:05
tags: [""]
categories:
description: "Explanation of transformation from one coordinate frame to
another."
---

Let $$\{B\}$$ be a [ [Coordinate Frame] ]( {% post_url
2022-02-21-coordinate_frame %} ) in frame $$\{A\}$$. The following kinds of
transformations are possible between $$\{B\}$$ and $$\{A\}$$.

## Translation

If $$\{B\}$$ has the same [ [Orientation] ]( {% post_url
2022-02-21-orientation %} ) as $$\{A\}$$ but the origin of
$$\{B\}$$ is at a distance of $$^AP_{Borg}$$ then if $$^BP$$ is a vector in
$$\{B\}$$, it can be represented in terms of $$\{A\}$$ as $$^AP=^BP +
^AP_{Borg}$$.

## Rotation

If $$\{B\}$$ has a different orientation than $$\{A\}$$ but their origin are the
same, then we can represent the points in $$\{B\}$$ in terms of $$\{A\}$$ by
multiplying them with the [ [Rotation Matrix] ]( {% post_url
2022-02-21-rotation_matrix %} ) that describes the orientation of $$\{B\}$$ with
respect to $$\{A\}$$.

## Combined transform

If $$\{B\}$$ has it's origin at $$^AP_{Borg}$$ and it has a orientation with
respect to $$\{A\}$$ given by $$^A_BR$$, then points in $$\{B\}$$ can be
represented in terms of $$\{A\}$$ by using the homogeneous transformation matrix
as

$$ \begin{bmatrix}^AP \\ 1 \end{bmatrix} =  \begin{bmatrix} ^A_BR && ^AP_{Borg}
\\ 0 && 1 \end{bmatrix} \begin{bmatrix}^BP \\ 1 \end{bmatrix}$$

Since both the $$^A_BR$$ and $$^AP_{Borg}$$ are invertible, therefore we can say
that the homogeneous transformation matrix is also invertible. 

If we have a point $$^AP$$ in $$\{A\}$$ and we want to represent it in terms of $$\{B\}$$ as $$^BP$$,
then we need to multiply $$^AP$$ with $$\begin{bmatrix} ^A_BR && ^AP_{Borg}
\\ 0 && 1 \end{bmatrix}^{-1}$$.

$$
\begin{align*}
\begin{bmatrix}
^A_BR && ^AP_{Borg} \\
0 && 1
\end{bmatrix}^{-1} &= 
\begin{bmatrix} 
\begin{bmatrix}
I && ^AP_{Borg} \\
0 && 1
\end{bmatrix}
\begin{bmatrix}
^A_BR && 0 \\
0 && 1
\end{bmatrix}
\end{bmatrix}^{-1} \\

&= \begin{bmatrix}
^A_BR && 0 \\
0 && 1
\end{bmatrix}^{-1}
\begin{bmatrix}
I && ^AP_{Borg} \\
0 && 1
\end{bmatrix}^{-1} \\

&= \begin{bmatrix}
^A_BR^T && 0 \\
0 && 1
\end{bmatrix}
\begin{bmatrix}
I && -^AP_{Borg} \\
0 && 1
\end{bmatrix} \\

&= \begin{bmatrix}
^B_AR && 0 \\
0 && 1
\end{bmatrix}
\begin{bmatrix}
I && -^AP_{Borg} \\
0 && 1
\end{bmatrix} \\

&= \begin{bmatrix}
^B_AR && -^B_AR ^AP_{Borg} \\
0 && 1
\end{bmatrix}
\end{align*}
$$

We know that $$^A_BR^{-1}$$ $$=$$ $$^A_BR^T$$ and $$^AP_{Borg}^{-1} = -^AP_{Borg}$$

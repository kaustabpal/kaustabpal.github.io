---
layout: post
title: "Coordinate Frame Transform"
permalink: "coordinate_transform"
date: 2022-02-21 20:59
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

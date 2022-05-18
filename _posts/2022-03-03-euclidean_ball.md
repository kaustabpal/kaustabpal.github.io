---
layout: post
title: "Euclidean Ball"
permalink: "euclidean_ball"
idate: 2022-03-05 11:08
date: 2022-03-05 11:08
tags: [""]
categories:
description: ""
---

{:class="table-of-content"}
* TOC 
{:toc}

Euclidean Balls $$B(x_c, r)$$ are sets of the form $$\{x \mid \lVert x-x_0
\rVert_{2} \leq r \}$$. Here $$x_c$$ is the center of the ball and $$x, x_c \in
R^n$$ and $$r \in R$$ and $$r \gt 0$$. They can also be represented as $$B(x_c,
r) = \{ x_c + ru  \mid \lVert u \rVert _2 \leq 1 \}$$. Here $$x_c \in R^n$$ is the
center of the ball, $$r \in R$$ is the radius of the ball and $$u \in R^n$$ is a
vector of unit length such that $$ru$$ will give us the vector of length $$r$$.

Euclidean balls are convex sets.

**Proof that euclidean balls are convex:**

Let $$x_1$$ and $$x_2$$ belong to the set $$B(x_c, r)$$. Let $$x_3$$ be a point
such that $$x_3 = \theta x_1 + (1-\theta) x_2$$. $$B(x_c, r)$$ will be convex if
$$x_3$$ also belongs to $$B(x_c, r)$$, i.e $$ \lVert x_3 - x_c \rVert _2 \leq
r$$.

$$
\begin{align*}
\lVert x_3 - x_c \rVert _2 &= \lVert \theta x_1 + (1-\theta) x_2 - x_c \rVert _2 \\
&\leq \theta \lVert x_1 - x_c \rVert _2 + (1-\theta) \lVert x_2 - x_c \rVert _2
\\
&\leq \theta r + r - \theta r \\
&\leq r
\end{align*}
$$

Since $$x_3 \leq r$$ we can conclude that it also lies in the euclidean ball.
Thus we can say that the [ [Convex Combination] ]( {% post_url
2022-02-19-convex_combination %} ) of any points in the euclidean ball will also
lie in the euclidean ball. Therefore we can say that the euclidean ball is a
[ [Convex Set] ]( {% post_url 2022-02-19-convex_set %} ).

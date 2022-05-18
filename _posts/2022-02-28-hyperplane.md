---
layout: post
title: "Hyperplane"
permalink: "hyperplane"
idate: 2022-02-28 23:32
date: 2022-02-28 23:32
tags: [""]
categories:
description: ""
---

{:class="table-of-content"}
* TOC 
{:toc}

A hyperplane is a set of the form $$\{x \mid a^Tx=b\}$$ such that $$a \in R^n$$
and $$a \neq 0 $$ and $$b \in R$$. In $$R^2$$ the hyperplane is a line. In
$$R^3$$ the hyperplane is a plane. 

Hyperplanes are not vector spaces unless they pass through the origin. A
hyperplane also divides a vector space $$R^n$$ into two [ [Halfspace] ]( {% post_url 2022-02-28-halfspace %} ).

A hyperplanes is [ [Convex Set]]( {% post_url 2022-02-19-convex_set %} ). 

**Proof:**

Let $$x_1$$ and $$x_2$$ be two points in the hyperplane.
Then $$a^Tx_1=b$$ and $$a^Tx_2=b$$. Now let $$c = \theta x_1 + (1-\theta) x_2$$.
For $$c$$ to lie in the hyperplane, the condition $$a^Tc=b$$ must be satisfied.

Now 

$$\begin{align*}
a^Tc &= a^T(\theta x_1 + (1-\theta) x_2)\\
&= \theta a^T x_1 + (1-\theta) a^T x_2 \\
&= \theta b + (1-\theta) b \\ 
&= \theta b + b - \theta b \\
&= b \\ 
\implies a^Tc &= b
\end{align*}
$$

Since the condition $$a^Tc=b$$ is satisfied, we can say that the [ [Convex Combination] ]( {% post_url 2022-02-19-convex_combination %} ) 
of any two points in the hyperplane also lies in the hyperplane. Hence we prove
that hyperplanes are convex sets.



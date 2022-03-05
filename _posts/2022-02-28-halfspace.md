---
layout: post
title: "Halfspace"
permalink: "halfspace"
date: 2022-03-03 22:50
tags: [""]
categories:
description: ""
---

{:class="table-of-content"}
* TOC 
{:toc}

A halfspace is a set of the form $$\{ x \mid a^Tx \leq b\}$$ such that $$a \in
R^n$$ and $$a \neq 0 $$ and $$b \in R$$. A halfspace is a [ [Convex Set] ]( {%
post_url 2022-02-19-convex_set %} ) but it isn't an [ [Affine Set] ]( {%
post_url 2022-02-19-affine_set %} ).

**Proof that a Halfspace is a Convex Set:**

Let $$x_1$$ and $$x_2$$ be two points in the halfspace. Then $$a^Tx_1 \leq b$$
and $$a^Tx_2 \leq b$$. Now let $$c = \theta x_1 + (1-\theta) x_2$$. For $$c$$ to
lie in the halfspace, the condition $$a^Tc \leq b$$ must be satisfied.

Now 

$$\begin{align*} a^Tc &= a^T(\theta x_1 + (1-\theta) x_2)\\ &= \theta a^T x_1 +
(1-\theta) a^T x_2 \\ &\leq \theta b + (1-\theta) b \\ &\leq \theta b + b -
\theta b \\ &\leq b \\ \implies a^Tc &\leq b \end{align*} $$

Since the condition $$a^Tc \leq b$$ is satisfied, we can say that the [ [Convex
Combination] ]( {% post_url 2022-02-19-convex_combination %} ) of any two points
in the halfspace also lies in the halfspace. Hence we prove that halfspaces are
convex sets.

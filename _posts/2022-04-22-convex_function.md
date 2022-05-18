---
layout: post
title: "Convex Function"
permalink: "convex_function"
idate: 2022-04-22 18
date: 2022-04-23 12:17
tags: [""]
category:
description: ""
---

{:class="table-of-content"}
* TOC 
{:toc}

A function is called a convex function if the domain of the function is a convex
set and if for all $$x,y \in \boldsymbol{dom} f$$ and $$\theta$$ with $$0 \leq \theta \leq
1$$, we have 

$$ f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y)$$

Geometrically the above inequality means that the line segment joining the points $$(x, f(x))$$
and $$(y, f(y))$$ must lie above the graph of the function $$f$$. 

The function $$f$$ is called strictly convex if there is a strict inequality in
the above function whenever $$x \neq y$$ and $$0 < \theta <  1$$.

The function $$f$$ is called concave if $$-f$$ is a convex function. Similarly
$$f$$ is strictly concave if $$-f$$ is strictly convex.

In the above equation if there is always a equality, then it is called an
[ [Affine Function] ]( {% post_url 2022-04-13-affine_function %} ).

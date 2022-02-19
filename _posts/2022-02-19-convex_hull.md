---
layout: post
title: "Convex Hull"
permalink: "convex_hull"
date: 2022-02-19
tags: [""]
categories:
---


The convex hull of the set $$C$$ is the set of all the [ [Convex Combination] ](
{% post_url 2022-02-19-convex_combination %} ) of the points in $$C$$. It is
denoted as $$conv C$$ and is always a [ [Connvex Set] ]( {% post_url
2022-02-19-convex_set %} ). The convex hull of a set is the smallest 
convex set containing that set. If $$B$$ is a convex set containing $$C$$, then
$$conv C \subseteq B$$.

$$conv C = \{ \theta_1 x_1 + \dots \theta_k x_k \mid x_i \in C, \theta_i \geq 0,
i = 1, \dots, k, \theta_1 + \dots + \theta_k =1 \}$$

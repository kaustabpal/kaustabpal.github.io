---
layout: post
title: "Project Euler: Problem 1"
permalink: "project_euler_1"
date: 2022-02-26
moddate: 2022-02-26 19:41
category: top
description: ""
---

## Question

If we list all the natural numbers below 10 that are multiples of 3 or 5, we get
3, 5, 6 and 9. The sum of these multiples is 23. Find the sum of all the
multiples of 3 or 5 below 1000.

## Solution

Multiples of $$3$$ are $$3,6,9,\dots,3i$$ such that $$3i \leq N$$ or $$i \leq
N/3$$. Thus we can write the sum of the multiples of $$3$$ as $$S_{3} = 3\times
(1+2+\dots +N/3)$$ or $$S_{3} = 3 \times (N/3)*(N/3+1)/2$$. Similarly we can
write the sum of the multiples of $$5$$ as $$S_{5} = 5 \times (N/5)*(N/5+1)/2$$.

$$S_3$$ and $$S_5$$ will give us the sum of the multiples of $$3$$ and $$5$$,
but there will be common multiples for $$3$$ and $$5$$ which will be added twice
in the sum. We need to subtract those common multiples from the sum to get the
correct answer. To do this we find the [ [Least Common Multiple] ]( {% post_url
2022-02-26-least_commom_multiple %} ) of the two numbers which is $$15$$. Then
the sum of the common multiples of $$3$$ and $$5$$ will be $$S_{15} = 15 \times
(N/15)*(N/15+1)/2$$.

We can now get the final sum as $$S = S_3 + S_5 - S_{15}$$.

Find the code [ [here] ](
https://github.com/kaustabpal/project_euler/blob/master/problem_1/problem_1.cpp
).

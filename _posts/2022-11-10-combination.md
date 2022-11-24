---
layout: post
title: "Combination"
permalink: "combination"
idate: 2022-11-10 15:58 # Post creation date
date: 2022-11-10 15:58 # Post update date
tags: ["probability"]
category:
description: ""
---

{:class="table-of-content"}
* TOC 
{:toc}


Combination tells us the number of ways we can create a subset of $$k$$ elements
given a set of $$N$$ elements.

Suppose we have a set of $$N$$ elements. From this set we want to create a
subset of $$k$$ elements. It is given by the formula $${N \choose k} =
\frac{N!}{K! (N-k)!}$$. This is also called the binomial coefficient and it
tells us how many $$k$$ element subsets are there in a set containing $$N$$
elements.

**Example 1:**

In an experiment, we toss a coin $$N$$ number of times. Out of $$N$$ tosses what
is the probability of $$k$$ heads.

Let the probability of heads be $$P(H) = p$$.

Let the probability of tails be $$P(T) = (1-p)$$.

Now let's say that the result of an experiment where $$N=6$$ and $$k=4$$ be
HHTTHTH.

$$
\begin{align}
P(HHTTHTH) &= p.p.(1-p).(1-p).p.(1-p).p \\
&= p^4 (1-p)2
\end{align}
$$

For any combination of $$4$$ heads, the probability will be $$p^4(1-p)^2$$

Therefore 

$$
\begin{align}
P(k=4) &= (\text{No. of k heads}) \times p^k(1-p)^{N-k} \\
&= {N \choose k}p^k(1-p)^{N-k}
\end{align}
$$

**Example 2:**

Event B: $$3$$ out of $$10$$ tosses were heads. 

Given that event $$B$$ has occured, what is the conditional probability that
event A: the first two tosses were heads has occured?

Number of elements in $$B$$ = $${10 \choose 3}$$

Probability of $$B$$ occuring is $$P(B) = {10 \choose 3}p^3(1-p)^{7}$$

In event $$A$$, the first two tosses were heads. Therefore the number of
elements in the event $$\mid A \cap B \mid$$ is the number of choices for the
third head since we already know that the first two tosses were heads. Since the
third toss can be in any of the remaining $$8$$ positions, $$\mid A \cap B \mid
= 8$$

Therefore the probability that the first two tosses are heads in the event $$B$$
is $$\frac{\mid A \cap B \mid}{\mid B \mid} = \frac{8}{10 \choose 3}$$


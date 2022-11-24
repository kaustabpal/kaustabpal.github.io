---
layout: post
title: "Permutation"
permalink: "permutation"
idate: 2022-11-04 10:34 # Post creation date
date: 2022-11-04 10:34 # Post update date
tags: ["probability"]
category:
description: ""
---

{:class="table-of-content"}
* TOC 
{:toc}

If a set contains $$N$$ elements, and we want to know what are the number of
ways we can order them, then there are $$N$$ choices for the first element,
$$N-1$$ choices for the second element and so on. We can write them as $$N
\times (N-1) \times (N-2) \times \dots \times 1 = N!$$. **Permutation** tells us
the number of ways we can order the set of $$N$$ elements.

If there are $$N$$ elements in a set and we want to know the number of subsets
we can make, then we have $$2$$ choices for the first element, i.e. we can
either keep it in the subset or won't keep it in the subset; $$2$$ choices for
the second element and so on for all the $$N$$ elements. Thus we will have
$$2^N$$ subsets from a set containing $$N$$ elements.

**Example 1**

What is the probability that six rolls of a six sided die all give different
numbers?

Total number of possible outcomes for $$6$$ rolls of a $$6$$ sided dice are
$$6^6$$, $$6$$ for the first, $$6$$ for the second and so on.

Number of elements in the event: $$6$$ options for the first roll, $$5$$ options
for the socond roll and so on. So $$6 \times 5 \times 4 \times \dots \times 1 =
6!$$

Therefore the probability of $$6$$ rolls of a $$6$$ sided dice all giving
different outcome is: $$\frac{6!}{6^6}$$

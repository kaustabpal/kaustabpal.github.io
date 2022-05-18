---
layout: post
title: "Greatest Common Divisor"
permalink: "gcd"
idate: 2022-02-26 00:23
date: 2022-02-26 00:23
tags: [""]
categories:
description: ""
---

The Greatest Common Divisor of a group of numbers is the greatest positive
integer that can divide all the numbers in the group without a reminder. 

## Euclid's algorithm to find GCD

```python
def gcd(a, b):
	if(a == 0):
		return b
	return gcd(b%a, a)
```

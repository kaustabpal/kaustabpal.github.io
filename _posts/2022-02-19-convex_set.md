--- 
layout: post 
title: "Convex Set" 
permalink: "convex_set"
date: 2022-02-28 23:25
tags: [""] 
categories: 
---

{:class="table-of-content"}
* TOC 
{:toc}

A set $$C$$ is called a convex set if it contains all the [ [Convex Combination]
]( {% post_url 2022-02-19-convex_combination %} ) of the points in that set.
Given a set $$C$$, we can make it convex by taking the [ [Convex Hull] ]( {%
post_url 2022-02-19-convex_hull %} ) of the set.

Examples of Convex Sets:
1. Empty set $$\{\emptyset \}$$ 
2. Singleton (A set containing a single point)
3. The whole $$R^n$$ space.
4. [ [Hyperplane] ]( {% post_url 2022-02-28-hyperplane %} )
5. [ [Halfspace] ]( {% post_url 2022-02-28-halfspace %} )
6. [ [Ellipsoid] ]( {% post_url 2022-03-05-ellipsoid %} )

Operations that preserve convexity:
1. Intersection: If two sets $$S_1$$ and $$S_2$$ are convex, then $$S_1 \cap
   S_2$$ is convex.
   
   **Proof:**
   
   Let $$x_1, x_2 \in S_1 \cap S_2$$. This means that $$x_1$$ and $$x_2 \in
   S_1$$ and $$x_1$$ and $$x_2 \in S_2$$. Therefore $$\theta x_1 + (1-\theta)
   x_2 \in S_1$$ since $$S_1$$ is convex. Similarly $$\theta x_1 + (1-\theta)
   x_2 \in S_2$$ since $$S_2$$ is convex. Therefore we can conclude that
   $$\theta x_1 + (1-\theta) x_2 \in S_1\cap S_2$$. Therefore we can say $$S_1
   \cap S_2$$ is convex.

   Vector subspaces, [ [Affine Set]s ]( {% post_url 2022-02-19-affine_set %} )
   and [ [Convex Cone]s ]( {% post_url 2022-02-19-convex_cone %} ) are also
   closed under arbitrary number of intersections. An exaple of this is a [
   [Polyhedron] ]( {% post_url 2022-03-05-polyhedron %} ).

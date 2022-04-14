--- 
layout: post 
title: "Panoptic Segmentation" 
permalink: "panoptic_segmentation"
date: 2022-02-14
moddate: 2022-02-14
tags: ["vision"] 
categories: notes
---
{:class="table-of-content"}
* TOC 
{:toc}

## Explanation

Panoptic segmentation unifies Instance segmentation and Semantic segmentation.
Here each pixel in the image is given a semantic label and an instance id.

Let there be a predetermined set of semantic labels $$L = \{ 0, 1, \dots ,
L-1\}$$. $$L$$ is a union of the set of labels belonging to **things** like cars,
cycle, pedestrians etc and the set of labels belonging to **stuffs** like sky, road,
grass etc, i.e. $$L = L^{t} \cup L^{st}$$. The task of panoptic segmentation requires
the algorithm to map each pixel $$i$$ in the image to a pair $$(l_i, z_i) \in L
\times \mathbb{N}$$. Here $$l_i$$ represents a label from set $$L$$ whereas
$$z_i$$ represents the instance number of that label. If a pixel is labelled
with $$l_i \in L^{st}$$ then we don't consider that pixel's instance number
$$z_i$$. The pixels belonging to an object in the scene will have the same
$$(l_i, z_i)$$ pair.  

## Important papers

[ [Panoptic Segmentation] ]( https://arxiv.org/abs/1801.00868 )
from FAIR labs gives a good overview of
the task along with it's challenges, datasets available and a new metric. 


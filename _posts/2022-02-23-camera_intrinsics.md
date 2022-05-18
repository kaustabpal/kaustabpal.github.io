---
layout: post
title: "Camera Intrinsics"
permalink: "camera_intrinsics"
idate: 2022-02-23 11:06
date: 2022-02-23 11:06
tags: [""]
categories:
description: ""
---

Given a point $$^kP$$ in the camera coordinate system $$\{k\}$$, the camera
intrinsic matrix projects that point to the sensor plane. The projected point is
in the homogeneous coordinate system. During this projection we lose imformation
as we are going from $$3d$$ to $$2d$$. Because of this reason, the camera
intrinsic matrix is non-invertible. 

The camera intrinsic matrix is also called the calibration matrix and is defined
as 

$$ k = \begin{bmatrix} c && cs && x_H \\ 0 && c(1+m) && y_H \\ 0 && 0 && 1
\end{bmatrix} $$

Here $$c$$ is the distance of the image plane from the camera, $$s$$ is the
sheer, $$m$$ is the scale difference between $$x$$ and $$y$$ of the sensor and
$$x_H$$ and $$y_H$$ are the coordinates of the [ [Principle Point] ]( {%
post_url 2022-02-23-principal_point %} ). Generally in a digital camera,
$$s=0$$.

## Todo
* Prove how we got the intrinsic matrix.

## Reference

*	[ [Camera Parameters: Extrinsics and Intrinsics] by Cyrill Stachniss ](
	https://www.youtube.com/watch?v=uHApDqH-8UE&t=1070s )

---
layout: post
title: "Camera Extrinsics"
permalink: "camera_extrinsics"
idate: 2022-02-23 11:04
date: 2022-02-23 11:04
tags: [""]
categories:
---

Camera Extrinsics are used to represent a point in the world coordinate frame
$$\{o\}$$ in terms of the camera coordinate frame$$\{k\}$$. To get the camera
extrinsics matrix we need the [ [Orientation] ]( {% post_url
2022-02-21-orientation %} ) of the camera with respect to $$\{o\}$$ given as
$$^o_kR$$ and the 
[ [Position] ]( {% post_url 2022-02-21-position_vector %} ) of the
origin of $$\{k\}$$ with respect to $$\{o\}$$ given as $$^oP_{korg}$$. Therefore the
transformation matrix from world to camera frame will be given as 

$$^kH_o = \begin{bmatrix}
^o_kR^T && -^o_kR^{T}{^oP_{korg}} \\
0 && 1
\end{bmatrix}
$$

See [ [Coordinate Frame Transform] ]( {% post_url
2022-02-21-coordinate_frame_transform %} ) for proof.

The camera extrinsic matrix changes as the camera moves around in the world
because the orientation and the position of the origin of the camera center with
respect to the world will also change. 

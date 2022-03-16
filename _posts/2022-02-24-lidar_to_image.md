---
layout: post
title: "Lidar to Camera Projection"
permalink: "lidar_to_camera"
date: 2022-02-25 11:33
tags: [""]
categories:
description: "How to project lidar points to an image."
---

To project lidar points to an image, we first need to represent the lidar points
from the lidar [ [Coordinate Frame] ]( {% post_url 2022-02-21-coordinate_frame
%} ) to the camera coordinate frame. We can do this by projecting the lidar
points to the car coordinate frame and then projecting the points from the car
coordinate frame to the camera coordinate frame. Both the camera and lidar
coordinate systems have their $$x$$ axis in the front, the $$y$$ axis to the
left and the $$z$$ axis pointing up. However in the camera sensor plane the
$$x$$ axis is pointing to the right, the $$y$$ axis is pointing to the down and
the $$z$$ axis is pointing to the front. Therefore we need to represent the
points in the camera coordinate frame in terms of the sensor coordinate frame.
We do this by rotating the points in the camera frame first along the $$z$$ axis
by an angle of $$90^\circ $$ and then along the $$x$$ axis by an angle of
$$90^\circ $$. 
After the new points are in the sensor frame, we can multiply the points with the [
[Camera Intrinsics] ]( {% post_url 2022-02-23-camera_intrinsics %} ) matrix to
project the points in the sensor plane. The points are now in the [ [Homogeneous
Coordinates] ]( {% post_url 2021-12-29-homogeneous-coordinates %} ) system. We
can get the sensor plane coordinates of the points by dividing the $$x$$ and
$$y$$ coordinates of the points with the $$z$$ coordinates. After this, we
eliminate the points which lie outside the image dimensions and finally display
the points on the image.


## Todo

1. Code for lidar to camera transform.

---
layout: post
title: "Lidar to Camera Projection"
permalink: "lidar_to_camera"
date: 2022-02-24 22:00 
tags: [""]
categories:
description: "How to project lidar points to an image."
---

To project lidar points to an image, we first need to represent the lidar points
from the lidar [ [Coordinate Frame] ]( {% post_url 2022-02-21-coordinate_frame
%} ) to the camera coordinate frame. We can do this by projecting the lidar
points to the car coordinate frame and then projecting the points from the car coordinate frame to the
camera coordinate frame. After the points are in the camera frame, we can multiply the
points with the [ [Camera Intrinsics] ]( {% post_url
2022-02-23-camera_intrinsics %} ) matrix to project the points in the sensor
plane. The points are now in the [ [Homogeneous Coordinates] ]( {% post_url
2021-12-29-homogeneous-coordinates %} ) system. We can get the sensor plane
coordinates of the points by dividing the $$x$$ and $$y$$ coordinates of the
points with the $$z$$ coordinates. After this, we eliminate the points which lie
outside the image dimensions and finally display the points on the image.


## Todo


---
layout: post
title: "Orientation"
permalink: "orientation"
date: 2022-02-21 12:18
tags: [""]
categories:
---

Given a [ [Position Vector] ]( {% post_url 2022-02-21-position_vector %} ), we
can describe it's orientation as a coordinate system attached to that point.
This new coordinate system $$\{B\}$$ is relative to the reference coordinate
system $$\{A\}$$. Describing $$\{B\}$$ with respect to $$\{A\}$$ will tell us
the orientation of the point with respect to $$\{A\}$$. We can use a [ [Rotation
Matrix] ]( {% post_url 2022-02-21-rotation_matrix %} ) to describe the
orientation of $$\{B\}$$ with respect to $$\{A\}$$.

There are multiple representations of orientation. Some of them are:
1. X-Y-Z fixed angles: This describes the rotation along the x-axis, y-axis and
   z-axis simultaneously with angles $$\gamma$$, $$\beta$$ and $$\alpha$$
   respectively. The rotations are performed along the axis of the fixed
   coordinate system $$\{A\}$$. This is also referred to as roll, pitch and yaw.
   The rotation matrix for X-Y-Z representation of an orientation is given by
   $$R_{xyz} = R_z(\alpha) R_y(\beta) R_x(\gamma)$$.

2. Z-Y-X Euler angles: Rotate $$B$$ along $$^BZ$$, then along $$^BY$$ and
   finally around $$^BX$$. Here the rotations are performed along the axis of
   the moving coordinate system.

3. Z-Y-Z Euler angles: We first rotate along $$^BZ$$, then along $$^BY$$ and
   finally along $$^BZ$$. The rotations happen along the axes of the moving
   coordinate system.

---
layout: post
title: "MPC in Autonomous Vehicles"
permalink: "mpc_autonomous_vehicles"
idate: 2022-03-18 01:49
date: 2022-03-18 01:49
tags: [""]
categories:
description: "Application of model predictive controls in autonomous vehicles."
---

{:class="table-of-content"}
* TOC 
{:toc}

Model Predictive Controls are being extensively used for trajectory optimization
problems in autonomous vehicles. The reason being we can directly add temporal
and spatial collision avoidance condtraints along with comfort constraints
without needing to seperate lateral and longitudinal vehicle dynamics.

It also has some drawbacks:

1. Issue of recursive feasibility:

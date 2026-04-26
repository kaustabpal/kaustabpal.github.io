---
title: "How Autonomous Vehicles Fail"
date: 2026-04-18
draft: false
tags: ["autonomous-driving", "systems"]
---

## Overview

AV failures are rarely random. They cluster around predictable gaps between the training distribution and the real world. This note maps out the main failure modes.

## Perception Failures

### Distribution Shift

Models trained on data from San Francisco fail on snow-covered roads in Pittsburgh. The visual features — lane markings buried under snow, different road surface textures — differ enough to push predictions off-distribution.

The fix is not just more data. It's understanding *which* distributional shifts matter for safety and testing against them explicitly.

### Edge Cases at Object Boundaries

Detection models struggle at the boundary between known and unknown. A couch on a highway is unusual but not fundamentally harder to detect than a car. The problem is that confidence calibration breaks down — the model may be confidently wrong rather than appropriately uncertain.

## Planning Failures

### Long-Tail Scenarios

An AV trained on millions of highway miles will have seen very few incidents involving the wrong-way driver. But wrong-way drivers cause severe accidents. Planning systems need explicit handling for rare, high-severity scenarios.

### Reward Misspecification

Optimizing for smoothness and collision avoidance can produce a system that is overly conservative — stopping at intersections longer than necessary, refusing lane changes. The reward function shapes behavior as much as the model capacity does.

## System-Level Failures

### Sensor Fusion Bugs

The LiDAR says there is an object. The camera says there is not. What does the fusion layer do? In poorly designed systems, one modality wins by default. In well-designed systems, disagreement triggers uncertainty-aware behavior.

### Latency and Temporal Inconsistency

A sensor reading from 50ms ago is already stale at highway speeds. Object velocity estimation errors compound over time. Systems that do not account for temporal uncertainty in their state estimates will make planning decisions on objects that have already moved.

## What Good Failure Looks Like

The goal is not zero failures — it is *safe* failures. A system that fails by stopping in place is better than one that fails by speeding up. Designing the failure mode is as important as optimizing nominal performance.

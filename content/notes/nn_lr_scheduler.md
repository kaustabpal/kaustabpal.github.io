+++
date = '2025-12-10T01:02:37+05:30'
draft = true
title = 'Neural Network: Distributed Training'
math = true
+++

## Model Parallelism

Turns out to be bad idea as the Gpus will be waiting. This is called model parallelism. 


Another approach is to run parallel branches of the model across multiple GPUss. This is what AlexNet did. 

This is also inefficient as synchronization across GPUs is expensive and activations and grad activations need to be transferred between GPUs.


## Data Parallelism 

We replicate the model on each GPUs and each GPU runs a minibatch and computes the forward pass and the backward pass. Gradients are then averaged across all GPUs and the updates weights are then copied to each models.

Problem Need to train with large number of mini batches. 

If on a single GPU we use learning rate $\\alpha$ and a batch size of $N$, then on K gpus, we use lr of $k\\alpha$ and batch size of $kN$.



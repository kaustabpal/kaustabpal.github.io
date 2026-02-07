+++
date = '2025-12-09T19:35:44+05:30'
draft = false
title = 'Neural Network: Regularization'
math = true
+++

## L1

L1 is less common than L2 but is highly valuable when aiming for network compression.

L1 Term: 

$$R(W)= \\sum_{\\text{all layers i}}\\|W^{(i)}\\|_{1}$$

We minimize the sum of the absolute values of all individual weights.

Goal: To drive weights of unimportant connections to exactly zero.

Effect: L1 acts as a pruning mechanism. If a weight $w_{jk}$ in matrix $W^(i)$ is set to 0, that specific connection between neuron $j$ and neuron $k$ is effectively removed from the network.

Benefit: Results in a sparse network architecture. This is useful for deploying models on devices with limited memory or computational power, as the network has fewer active parameters.

As a reminder, the geometric reason for this behavior remains the same:

The L1 diamond constraint has sharp corners aligned with the axes, forcing the optimum solution to land right on the axis where some weights are zero. The L2 circular constraint smoothly shrinks weights towards zero, but never eliminates them entirely.


## L2

L2 regularization is also known as weight decay where we use $W^TW$ as the regularization term.

The goal is to prevent any single weight from becoming overly large. 

L2 is the default and most common regularization technique in Neural Networks.

L2 Term: $R(W)=\\sum_{\\text{all layers i}} \\|W^{(i)}\\|^{2}_{F}$

We minimize the sum of the squares of all individual weights $w_{jk}$ in all layers $W^{(i)}$.

Goal: To prevent any single weight from becoming overly large.

Effect: By keeping all weights small, the network's function remains "smooth," meaning that small changes in the input result in small changes in the output. This heavily reduces overfitting.

Benefit: Provides stability and better generalization, especially in deep networks where gradient explosions are a concern (as discussed in the $W^TW$ explanation). It is differentiable everywhere, making it easy to optimize.

1. The Geometric Interpretation: What is $W^TW$?

To understand the regularizer, you must look at the geometry of the weight matrix $W$. If we treat the columns of the weight matrix $W$ as individual feature vectors (or filters) $w_1$, $w_2$, $w_3$ and so on, then the matrix product $W^TW$ is the Gram Matrix (the matrix of all possible dot products).

Each entry at position (i,j) in $W^TW$ represents the dot product between two weight vectors:

(W^TW)_{ij} = w_i^Tw_j

The Diagonal elements (i=j): Represent the squared length (magnitude) of each vector.

The Off-diagonal elements (i\neq j): Represent the correlation (overlap) between different vectors.

2. Promoting Feature Diversity (Decorrelation)

The most common reason to use this regularizer is to force the neural network to learn diverse features.

If you minimize the off-diagonal elements of $W^TW$, you are forcing $w_i^Tw_j \approx 0$ for all distinct $i$ and $j$.

Without this: A network might learn two filters that look nearly identical (highly correlated). This is a waste of parameters and computation.

With this: The vectors are pushed to be perpendicular (orthogonal) to each other. This ensures that filter A is looking for something completely different than filter B.

3. Preventing Vanishing and Exploding Gradients

This is critical in very deep networks (like Deep ResNets) or Recurrent Neural Networks (RNNs).

When you forward-propagate a signal through many layers, you are effectively multiplying the input by W repeatedly.

If the eigenvalues of W are >1, the signal (and gradients) explode to infinity.

If the eigenvalues of W are <1, the signal (and gradients) vanish to zero.

The Solution: If we force $W^TW \approx I$ (the Identity matrix), then W becomes an Orthogonal Matrix. Orthogonal matrices are "isometric," meaning they rotate vectors without changing their length (magnitude).


This keeps the gradient norm preserved effectively throughout the network, allowing for much faster and more stable convergence in deep architectures

## Dropout

In each forward pass randomly set some neurons to 0. Probability of setting a neuron to 0 is given by the dropout parameter. This is only during training.

It forces the network to have redundant information or prevents co-adaptation of features. Learns robust information.

Another way to think about dropout is to think of it as a form of ensembling. Each forward pass is a different model. 

Dropot makes the output random. During test time we want a deterministic system. So how do we use dropout during test time? 

We use all the neurons but we rescale activations by the dropout parameter. This is to match the expected output of the network during training time. Another way to do this is by doing inverted dropout where we divide the activations by the dropout parameter.

Many of the modern architectures like GoogleNet and ResNet don't use dropout atall but rather use global average pooling.

Consider dropout for very very large fully connected networks. 

> **Future work:** Need to understand the derivation of why this is happening during test time from [Lecture 10: Training Neural Networks I](https://youtu.be/lGbQlr1Ts7w?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&t=3347) 

> Read: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

## Batch Normalization

Adds randomness during training time because the outputs of each elements in the batch depend on every other element in the batch because during training as the stats are computed on the mini-batch. Therefore during every epoch of the training as the mini-batch changes the stats also changes. 

For Resnet and later, often L2 and Batch normalization are the only regularization methods used.


## Data Augmentation

Not considered as a regularization method but also adds randomness during training time.


## Drop Connect

Sets weight to 0 during training time. 

Uses all connections during test time.

> Read: [Regularization of Neural Networks using DropConnect](https://proceedings.mlr.press/v28/wan13.pdf)

## Fractional Pooling

Randomize the size of the pooling operations during the forward pass. Some will have 3x3 and some will have 2x2.

> Read: [Fractional Max-Pooling](https://arxiv.org/abs/1412.6071)

## Stochastic Depth

If we have a 100 layer network, during training time we will use a subset of these blocks. During test time we will use all the blocks.

> Read: [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

## Cutout

Sets random image regions to $0$.

> Read: [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)

## Mixup

During training, blend images from two different classes like $40\\%$ cat and $60\\%$ dog.

> Read: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)







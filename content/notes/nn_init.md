+++
date = '2025-12-09T19:14:06+05:30'
draft = true
title = 'Neural Network Initialization'
math = true
+++


## Introduction

Weight initialization is one of the most critical yet often overlooked aspects of training neural networks. The way you initialize your network's weights can mean the difference between a model that converges quickly and one that struggles to learn at all. This post explores the fundamental question: **How do you initialize a neural network?**

Based on insights from Stanford's CS231n lecture on "Training Neural Networks I," we'll dive deep into different initialization strategies, their effects on training, and how they interact with various activation functions.

## The Constant Initialization Problem

### All Zeros: The Symmetry Trap

What happens if we initialize all weights to zero?

**Answer:** Complete failure. Here's why:

- All neurons in a layer compute the exact same output
- All neurons receive the exact same gradient during backpropagation
- Weights remain identical throughout training (symmetry is never broken)
- The network effectively behaves as if it has only one neuron per layer

**Verdict:** ❌ Never initialize to all zeros.

### All Ones (or Any Constant K)

What if we initialize all weights to 1, or any constant value K?

**Answer:** Same problem, different magnitude:

- Symmetry still exists - all neurons are identical
- Gradients are identical across neurons in the same layer
- No differentiation between neurons
- The network can't learn diverse features

**Verdict:** ❌ Constant initialization breaks the fundamental requirement of neural networks: **asymmetry**.

## Random Initialization: The Foundation

To break symmetry, we need randomness. But how much, and from what distribution?

### Gaussian Distribution

Initializing weights from a Gaussian (normal) distribution: `W ~ N(0, σ²)`

**Key Question:** What should σ (standard deviation) be?

**Too Small (σ → 0):**
- Activations become very small
- Gradients vanish as they backpropagate
- Deep networks fail to train

**Too Large (σ → $\infty$):**
- Activations saturate (especially with sigmoid/tanh)
- Gradients vanish in saturated regions
- Training stalls

**The Goldilocks Zone:** We need σ "just right" - this led to principled initialization schemes.

### Uniform Distribution

Initializing from a uniform distribution: `W ~ U(-a, a)`

Similar considerations apply:
- Range `a` must be carefully chosen
- Too small → vanishing activations
- Too large → saturation
- Often used in Xavier initialization

## Activation Functions and Initialization

The choice of activation function dramatically affects how we should initialize weights.

### Sigmoid and Tanh: The Vanishing Gradient Problem

**Sigmoid:** `σ(x) = 1 / (1 + e^(-x))`
**Tanh:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Problems:**
- Saturate for large positive/negative inputs
- Gradients approach zero in saturated regions
- Poor initialization → immediate saturation → vanishing gradients
- Deep networks become nearly impossible to train

**Solution:** Xavier/Glorot Initialization

### ReLU: A Game Changer

**ReLU:** `f(x) = max(0, x)`

**Advantages:**
- No saturation for positive values
- Gradients are either 0 or 1
- Mitigates vanishing gradient problem
- Faster training

**New Problem:** Dead ReLUs
- Neurons can "die" if they always output 0
- Requires careful initialization

**Solution:** He/MSRA Initialization

## Modern Initialization Schemes

### Xavier/Glorot Initialization

**For Sigmoid/Tanh Networks**

Designed to maintain variance of activations and gradients across layers.

**Formula:**
```
W ~ N(0, 1/n_in)  or  W ~ U(-√(1/n_in), √(1/n_in))
```

Where `n_in` is the number of input units.

**Intuition:**
- Keeps signal variance roughly constant
- Prevents explosion or vanishing of activations
- Works well for symmetric activation functions

### He/MSRA Initialization

**For ReLU Networks**

Accounts for the fact that ReLU kills half the neurons (sets negative values to 0).

**Formula:**
```
W ~ N(0, 2/n_in)
```

**Key Difference:** Factor of 2 compensates for ReLU zeroing out half the activations.

**Why it works:**
- ReLU cuts variance in half
- Multiplying by 2 compensates for this
- Maintains healthy activation variance through deep networks

## Practical Guidelines

### Choosing Initialization Based on Activation

| Activation Function | Recommended Initialization |
|---------------------|---------------------------|
| Sigmoid/Tanh | Xavier/Glorot |
| ReLU | He/MSRA |
| Leaky ReLU | He (with adjustment) |
| Linear | Xavier |

### Implementation Tips

**PyTorch:**
```python
# Xavier
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

# He
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

**TensorFlow/Keras:**
```python
# Xavier
tf.keras.initializers.GlorotUniform()
tf.keras.initializers.GlorotNormal()

# He
tf.keras.initializers.HeUniform()
tf.keras.initializers.HeNormal()
```

## Summary: The Initialization Decision Tree

1. **Never** use constant initialization (all zeros, all ones, etc.)
2. **Always** use random initialization to break symmetry
3. **Match** initialization to activation function:
   - Sigmoid/Tanh → Xavier/Glorot
   - ReLU/variants → He/MSRA
4. **Monitor** activation statistics during training
5. **Adjust** if you see vanishing/exploding gradients

## Conclusion

Proper weight initialization is not just a technical detail—it's a fundamental requirement for training deep neural networks effectively. The evolution from random initialization to Xavier and He schemes represents our growing understanding of how information flows through deep networks.

The key insight: **initialization must account for both network architecture (depth, width) and activation functions** to maintain healthy gradient flow during training.

---

*This post is based on Stanford's CS231n: Convolutional Neural Networks for Visual Recognition, [Lecture 10: Training Neural Networks I](https://www.youtube.com/watch?v=lGbQlr1Ts7w&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=10)*
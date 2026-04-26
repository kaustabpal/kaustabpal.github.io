---
title: "Understanding Attention Mechanisms"
date: 2026-04-10
draft: true
tags: ["deep-learning", "transformers"]
---

## Overview

Attention mechanisms are at the core of modern deep learning architectures. This note breaks down how they work from first principles.

## The Core Idea

Before transformers, sequence models processed tokens one at a time. Attention lets every token look at every other token simultaneously, computing relevance scores between positions.

Given queries `Q`, keys `K`, and values `V`:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

The `sqrt(d_k)` scaling prevents the dot products from growing too large, which would push softmax into regions with very small gradients.

## Multi-Head Attention

Running attention once uses all of the model's capacity on a single representation. Multi-head attention splits the embedding into `h` heads, runs attention independently on each, then concatenates:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

Different heads specialize. Some attend to syntactic relationships, others to semantic ones.

## Self-Attention vs Cross-Attention

**Self-attention**: Q, K, V all come from the same sequence. Each token attends to all other tokens in the same sequence.

**Cross-attention**: Q comes from one sequence, K and V from another. Used in encoder-decoder architectures to attend to encoder output.

## Computational Cost

Self-attention is O(n²) in sequence length. For a sequence of length 1024, that's ~1M attention pairs. This is why long-context models are expensive and why efficient attention variants (sparse, linear) matter.

## Where It Fails

Attention has no built-in notion of position. Without positional encodings (absolute, relative, RoPE, ALiBi), the model treats sequences as unordered bags of tokens. The choice of positional encoding significantly affects how well the model extrapolates to longer sequences than those seen during training.

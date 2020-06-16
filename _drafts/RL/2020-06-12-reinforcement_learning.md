---
layout: post
title: "An essay on reinforcement learning"
excerpt: "An overview of the concepts of Linear Algebra."

---
{: class="table-of-content"}
* TOC
{:toc}
# Markov Decision Process

![Markdov Decision Process](/assets/img/RL/mdp.png)

*Fig 1. A MDP is an agent. The agent at time $$t$$ chooses an action $$a_t$$. As a consequence, the state of the environment will change and a reward will be given to the agent at time $$t+1$$.*

In a typical RL system, an agent acts in an environment. Due to the agent's action $$a_t$$ at time $$t$$, the environment's state changes from $$s_t$$ to $$s_{t+1}$$. Due to this change of state the environment gives a reward $$r_{t+1}$$ to the agent at time $$t+1$$. A reward function tell us that if we start in state $$s$$ and take action $$a$$ what is the expected reward we can get. It is given by $$R(s,a)=E[r_{t+1}\mid s,a]$$.

The goal of the agent will be to choose actions that will maximize the cumulative rewards in the long run. to maximize the cumulative reward in the long run, the agent needs to be in states that will lead to the max cumulative reward. This is determined by the value of the state. The value of the state tells us what is the expected cumulative reward we can expect to collect starting from that state.

$$V(s)=E[R_{t+1}+\gamma V(s_{t+1}) \mid s]$$

RL algos that chooses actions that leads to the highest values in the next state are called value based algorithms.

An agent can also choose actions based on a policy. These agent's are called policy based agents. A policy is a function that takes the present state as input and gives the action to perform as an output. A policy can be deterministic, i.e. given a state it can output only one action or stochastic, i.e. given a state the policy will output a probability distribution of the possible actions. A policy is represented by $$\pi$$.

- Deterministic policy: $$\pi(s)=a$$ or $$ \pi(a\mid s)= \begin{cases}
  1 & \text{if a will be taken }\\    
  0 & \text{if a will not be taken}    
\end{cases}$$

- Stochastic policy: $$\pi(a\mid s) = p(a\mid s)=$$ some value between $$0$$ and $$1$$.

After taking an action, depending on the environment the agent will land either on it's desired state or land in a different state. This is modeled by transition functions. Given a state and an action, the transition function will tell us the probability of landing in a particular state. It is defined by $$P(s'\mid s,a)$$. A transition matrix tells us the transition probabilities from all states $$s$$ to all successor states $$s'$$ when we take a particular action. It is defined by $$P^a$$

$$P^a = \begin{bmatrix}
P_{11}^a & P_{12}^a & \dots P_{1n}^a\\
\vdots & \vdots & \vdots\\
P_{n1}^a & P_{n2}^a & \dots P_{nn}^a
\end{bmatrix}$$

If we are given an MDP, we need to find the optimal policy $$\pi^{\star}$$ such that if the agent follows $$\pi^{\star}$$, the expected sum of rewards from t=0 to t=H will be maximised.

There are two methods to find the policy $$\pi^{\star}$$

- Value based iteration

- Policy based iteration

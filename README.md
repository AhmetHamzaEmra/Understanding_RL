# Understanding Reinforcement Learning 

Deep Reinforcement Learning requires good amount of intuation in both deep learning and reinforcement learning. Even though, theorical part is not that hard to understand, it definitely makes it harder to understand the messy codes. Lets try to change that :D 

---

#### Section 1

## OpenAI gym

Since we are working with OpenAI's GYM alot, lets have better intuation about it! 

[OpenAi Gym enviroments](https://github.com/AhmetHamzaEmra/Understanding_RL/blob/master/openai_gym/Understanding%20Gym%20enviroments.ipynb)

---

#### Section 2 

## Policy Gradient menthods 

In supervised learning, it is simple to create a system that can easily map inputs X into outputs Y since there is a dataset which contains all input and output examples. On the other hand in Reinforcement learning, there are no datasets which contain examples just like datasets in supervised learning. Using Policy gradient is one way to solve this problem. The hole idea relly on encouraging the actions with good reward and discouraging the actions with bad reward. The general formula is minimizing the   <u>*log(p(y | x))  A*</u>  loss. In here A represent Adventage and for most vanilla version we can use discounted rewards. 

* [Policy gradient from scratch](https://github.com/AhmetHamzaEmra/Understanding_RL/blob/master/policy_gradient/Understanding%20Policy%20Gradient%20.ipynb)

* [Tensorflow v1.*]()

  ![](https://raw.githubusercontent.com/AhmetHamzaEmra/Understanding_RL/master/policy_gradient/lunar.gif)

* [Tensorflow v2.0]() 

* Pythorch *Coming Soon*



Extra resources:

1. [My blog post](https://medium.com/@hamza.emra/reinforcement-learning-with-tensorflow-2-0-cca33fead626)
2. [Pong from pixels](http://karpathy.github.io/2016/05/31/rl/)
3. Hands-On Machine Learning with Scikit-Learn and TensorFlow Chapter 16
4. [Policy Gradients Pieter Abbeel lecture ](https://www.youtube.com/watch?v=S_gwYj1Q-44)
5. [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
6. [Better Exploration with Parameter Noise](https://blog.openai.com/better-exploration-with-parameter-noise/)

---

#### Section 3

## Deep Q Networks

Before we start with DQN lets talk about Q function first. Q(s,a)​ is  a function that maps given ​s (state) and a(action) pair to expected total reward untile the terminal state. It is basicaly how much reward we are gonna gate if we act with action a in state s. The reason we combine this idea with NN is it is almost imposible to find  ​*q values* for all states in environment. 

 





1. [DQN](DeepQ)
2. [Baselines](Baselines)


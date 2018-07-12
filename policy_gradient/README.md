# Understanding Policy Gradient 



In supervised learning, it is simple to create a system that can easily map inputs X into outputs Y since there is a dataset which contains all input and output examples. On the other hand in Reinforcement learning, there are no datasets which contain examples just like datasets in supervised learning. Using Policy gradient is one way to solve this problem.



How it works:

1. Create a network (in our case it is two layered feedforward network)
2. Compute the forward pass and predict an action
3. Assume that action is correct and compute gradient with the backward pass
4. Get the reward for that action
5. Multiply rewards with gradient and make the update



So if the reward is positive that means we did something good and make update accordingly, but if the reward is negative that means discourage that action for that state.



Sample Code:

[Policy Gradient from scratch](https://github.com/AhmetHamzaEmra/Understanding_RL/blob/master/policy_gradient/Understanding%20Policy%20Gradient%20.ipynb)

[Policy Gradient with Tensorflow](https://github.com/AhmetHamzaEmra/Understanding_RL/blob/master/policy_gradient/PG_Tensorflow.ipynb)

![](https://github.com/AhmetHamzaEmra/Understanding_RL/blob/master/policy_gradient/lunar.gif)



Readings :

1. [Deep Reinforcement Learning: Pong from Pixels by Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/)

2. Hands-On Machine Learning with Scikit-Learn and TensorFlow Chapter 16

3. [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

4. [Better Exploration with Parameter Noise](https://blog.openai.com/better-exploration-with-parameter-noise/)

   

    

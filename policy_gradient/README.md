# Understanding Policy Gradient 



In supervised learning, it is simple to create a system that can easily map inputs X into outputs Y since there is a dataset which contains all input and output examples. On the other hand in Reinforcement learning, there are no datasets which contain examples just like datasets in supervised learning. Using Policy gradient is one way to solve this problem.



How it works:

1. Create a network (in our case it is two layered feedforward network)
2. Compute the forward pass and predict an action
3. Assume that action is correct and compute gradient with the backward pass
4. Get the reward for that action
5. Multiply it with rewards



So if the reward is positive that means we did something good and make update accordingly, but if the reward is negative that means discourage that action for that state.



Sample Code:

[Policy Gradient from scratch](Understanding Policy Gradient .ipynb)



Readings :

1. [Deep Reinforcement Learning: Pong from Pixels by Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/)

2. Hands-On Machine Learning with Scikit-Learn and TensorFlow Chapter 16

   

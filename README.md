# Understanding Reinforcement Learning 

Deep Reinforcement Learning requires good amount of intuation in both deep learning and reinforcement learning. Even though, theorical part is not that hard to understand, it definitely makes it harder to understand the messy codes. Lets try to change that :D 

---

#### Section 1

##OpenAI gym

Since we are working with OpenAI's GYM alot, lets have better intuation about it! [OpenAi Gym enviroments](https://github.com/AhmetHamzaEmra/Understanding_RL/blob/master/openai_gym/Understanding%20Gym%20enviroments.ipynb)

---

#### Section 2 

## Policy Gradient menthods 

In supervised learning, it is simple to create a system that can easily map inputs X into outputs Y since there is a dataset which contains all input and output examples. On the other hand in Reinforcement learning, there are no datasets which contain examples just like datasets in supervised learning. Using Policy gradient is one way to solve this problem. The hole idea relly on encouraging the actions with good reward and discouraging the actions with bad reward. The general way is minimizing the  \( \sum_i A_i \log p(y_i \mid x_i) \)



1. [DQN](DeepQ)
2. [Baselines](Baselines)


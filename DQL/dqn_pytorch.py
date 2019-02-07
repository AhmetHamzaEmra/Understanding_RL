import numpy as np 
import random
import gym
import numpy as np
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F

EPISODES = 1000

class DQN(nn.Module):
    def __init__(self, state_size,
    action_size, 
    epsilon=1.0,
    epsilon_decay=0.99, 
    epsilon_min= 0.01,
    gamma=0.9):

        super(DQN, self).__init__()

        hidden_1 = 20
        hidden_2 = 20
        self.action_size = action_size
        self.state_size = state_size

        self.epsilon=epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.memory = deque(maxlen=2000)

        self.fc1 = nn.Linear(state_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        self.fc3 = nn.Linear(hidden_2, action_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

    def sample_action(self, x):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return torch.argmax(self.forward(x))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
net = DQN(state_size, action_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
batch_size = 32


for e in range(EPISODES):
    state = env.reset()
    done = False
    state = np.reshape(state, [1, state_size])
    train_loss = 0.0
    net.train()
    episode_reward = 0 
    while not done:
        # sample an action 
        action = net.sample_action(state)
        # get the observation
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        episode_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        # save the observation into memory 
        net.remember(state, action, reward, next_state, done)
        state = next_state

        if len(net.memory) > batch_size:
            minibatch = random.sample(net.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                

                target = net.forward(state)
                print(target)
                if done:
                    target[0][action] = reward
                else:
                    t = net.forward()
                    target[0][action] = reward + net.gamma * np.amax(t)
                loss = criterion(state, target)
                loss.backward()
                optimizer.step()
            if net.epsilon > self.epsilon_min:
                net.epsilon *= net.epsilon_decay

    print("episonde: {}/{}, score: {}, e: {:.2}".format(e,EPISODES,episode_reward, net.epsilon ))

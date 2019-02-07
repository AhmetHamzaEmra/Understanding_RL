"""
Sample implementation of DQN 

Todo:

deque memory needs to be converted to torch matrix

"""


import numpy as np 
import random
import gym
import numpy as np
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F

EPISODES = 1000

class agent_network(nn.Module):
    def __init__(self, state_size, action_size):
        super(agent_network, self).__init__()
        hidden_1 = 20
        hidden_2 = 20
        self.fc1 = nn.Linear(state_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    



class DQN(nn.Module):
    def __init__(self, state_size,
    action_size, 
    epsilon=1.0,
    epsilon_decay=0.99, 
    epsilon_min= 0.01,
    gamma=0.9):
        super(DQN, self).__init__()

        self.action_size = action_size
        self.state_size = state_size

        self.epsilon=epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.memory = deque(maxlen=2000)
        self.model = agent_network(state_size, action_size)
        self.target_network = agent_network(state_size, action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)


    def sample_action(self, x):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return torch.argmax(self.model.forward(x))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def soft_update(self, model, target_model, tau):
        for target_param, model in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau*model.data + (1.0-tau)*target_param.data)

    def act(self, x):
        return torch.argmax(self.model.forward(x))


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size)
    batch_size = 32


    for e in range(EPISODES):

        state = env.reset()
        done = False
        state = np.reshape(state, [1, state_size])
        dqn.model.train()
        episode_reward = 0 
        while not done:
            # sample an action 
            action = dqn.sample_action(state)
            print(action)
            # get the observation
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            episode_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            # save the observation into memory 
            dqn.remember(state, action, episode_reward, next_state, done)
            state = next_state

            if len(dqn.memory) > batch_size:
                minibatch = random.sample(dqn.memory, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = dqn.model.forward(state)
                    if done:
                        target[0][action] = reward
                    else:
                        t = dqn.target_network.forward(next_state)[0]
                        target[0][action] = reward + dqn.gamma * np.amax(t)
                    loss = dqn.criterion(state, target)
                    loss.backward()
                    dqn.optimizer.step()
                if dqn.epsilon > self.epsilon_min:
                    dqn.epsilon *= dqn.epsilon_decay

        if e % 10 == 0:
            dqn.soft_update(dqn.model, dqn.target_network, tau=0.3)


        print("episonde: {}/{}, score: {}, e: {:.2}".format(e,EPISODES,episode_reward, dqn.epsilon ))

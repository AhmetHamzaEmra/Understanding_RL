"""
Sample implementation of DQN 
"""


import numpy as np 
import random
import gym
import numpy as np
from collections import namedtuple, deque
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  

EPISODES = 500
TESTCASE = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



class DQN(nn.Module):
    def __init__(self, state_size,
    action_size, 
    epsilon=1.0,
    epsilon_decay=0.999, 
    epsilon_min= 0.01,
    gamma=0.9):
        super(DQN, self).__init__()

        self.action_size = action_size
        self.state_size = state_size

        self.epsilon=epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.memory = ReplayBuffer(action_size, buffer_size=2000, batch_size=32)

        self.model = agent_network(state_size, action_size)
        self.target_network = agent_network(state_size, action_size)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


    def sample_action(self, x):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.forward(x).cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def soft_update(self, model, target_model, tau):
        for target_param, model in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau*model.data + (1.0-tau)*target_param.data)

    def train_step(self):

        states, actions, rewards, next_states, dones = self.memory.sample()

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.model(states).gather(1, actions)
        # Compute loss
        loss = self.criterion(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, x):
        return np.argmax(self.target_network.forward(x).cpu().data.numpy())

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)




"""
Test case for DQN pytroch!

"""
import numpy as np
import gym
import torch
from collections import deque
from dqn_pytorch import DQN

EPISODES = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
dqn = DQN(state_size, action_size)
batch_size = 32
scores = deque(maxlen=10)
best_mean = 0.0
### TRAINING ###
for e in range(EPISODES):

    state = env.reset()
    done = False
    state = np.reshape(state, [1, state_size])
    dqn.model.train()
    episode_reward = 0 
    for t in range(500):
        # sample an action 
        state = torch.from_numpy(state).float().to(device)
        action = dqn.sample_action(state)
        
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        episode_reward+=reward
        next_state = np.reshape(next_state, [1, state_size])
        # save the observation into memory 
        dqn.remember(state, action, reward, next_state, done)
        state = next_state
        


        if len(dqn.memory) > batch_size:
            dqn.train_step()
        
        if done:
            dqn.soft_update(tau=0.3)
            scores.append(episode_reward)
            if np.mean(scores) > best_mean:
                dqn.save_model("checkpoint.pth")
                best_mean = np.mean(scores)

            print("episonde: {}/{}, n_action: {}, mean score: {:.2}, epsilon: {:.2}".format(e,EPISODES, t, np.mean(scores),dqn.epsilon ))
            break
        


print("\nSaving the Model\n")
dqn.save_model("pole.pth")


env.close()

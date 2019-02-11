"""
Test case for DQN pytroch!
LunarLander-v2
"""
import numpy as np
import gym
import torch
from dqn_pytorch import DQN
from collections import deque
EPISODES = 100000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
dqn = DQN(state_size, action_size)
batch_size = 32
gamma = 0.9
scores = deque(maxlen=20)
print("Last checkpoint loaded!")
dqn.load_model("53lunar_checkpoint.pth")
best_mean = 40.0
### TRAINING ###

for e in range(EPISODES):

    state = env.reset()
    done = False
    state = np.reshape(state, [1, state_size])
    dqn.model.train()
    experience = []
    episode_memory = [] 
    episode_reward = 0 
    for t in range(1000):
        # sample an action 
        state = torch.from_numpy(state).float().to(device)
        action = dqn.sample_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward+=reward
        next_state = np.reshape(next_state, [1, state_size])
        current_memory = [state, action, reward, next_state, done]
        episode_memory.append(current_memory)

        state = next_state

        if len(dqn.memory) > batch_size:
            dqn.train_step()

        if done:
            dqn.soft_update(tau=0.3)
            
            for i, v in enumerate(episode_memory):
                for j in range(i+1, len(episode_memory)):
                    episode_memory[i][2] += gamma**(j-i) * episode_memory[j][2] 

            dqn.remember(episode_memory)

            scores.append(episode_reward)

            if np.mean(scores) > best_mean:
                best_mean = np.mean(scores)
                dqn.save_model(str(str(int(best_mean)) + "lunar_checkpoint.pth"))
            print("episonde: {}/{}, score: {}, e: {:.2}".format(e,EPISODES, episode_reward, dqn.epsilon ))
            break


env.close()


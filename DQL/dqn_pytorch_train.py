"""
Test case for DQN pytroch!

"""

import numpy as np
import gym
import torch
from dqn_pytorch import DQN

EPISODES = 2000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size)
    batch_size = 32
    scores = []
    ### TRAINING ###
    for e in range(EPISODES):

        state = env.reset()
        done = False
        state = np.reshape(state, [1, state_size])
        dqn.model.train()
        episode_reward = 0 
        while not done:
            # sample an action 
            state = torch.from_numpy(state).float().to(device)
            action = dqn.sample_action(state)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            # save the observation into memory 
            dqn.remember(state, action, reward, next_state, done)
            state = next_state

            if len(dqn.memory) > batch_size:
                dqn.train_step()

        if e % 10 == 0:
            dqn.soft_update(dqn.model, dqn.target_network, tau=0.75)
        scores.append(episode_reward)
        print("episonde: {}/{}, score: {}, e: {:.2}".format(e,EPISODES,episode_reward, dqn.epsilon ))


    print("\nSaving the Model\n")
    dqn.save_model("LunarLander.pth")


    env.close()

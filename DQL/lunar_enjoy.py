"""
Test cases for Lunarlander!

"""


import numpy as np 
import gym
import torch
from dqn_pytorch import DQN

TESTCASE = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size, epsilon_min=0.01)
    batch_size = 32
    scores = []
    dqn.load_model("61lunar_checkpoint.pth")

    ### TESTING ###
    print("\nTESTING\n")

    for t in range(TESTCASE):
        state = env.reset()
        done = False
        episode_reward = 0 
        while not done:
            env.render()
            # sample an action 
            state = np.reshape(state, [1, state_size])
            state = torch.from_numpy(state).float().to(device)
            action = dqn.act(state)
            # get the observation
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        print("Test case number {}, episode reward {}".format(t, episode_reward))

    env.close()

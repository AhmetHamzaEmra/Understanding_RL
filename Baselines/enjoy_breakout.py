import gym

from baselines import deepq
from baselines.common.atari_wrappers import make_atari, WarpFrame, FrameStack

def main():

    env = make_atari("BreakoutNoFrameskip-v0")
    env = WarpFrame(env)
    env = FrameStack(env, k=4)

    act = deepq.load("breakout_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
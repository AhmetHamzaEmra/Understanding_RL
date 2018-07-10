from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari, WarpFrame, FrameStack

def main():
    # create the game enviroment
    # To use make_atari from baselines name must contains "NoFrameskip" 
    env = make_atari("BreakoutNoFrameskip-v0")
    # Convert it to gray scale and resize it to 84x84
    env = WarpFrame(env)
    # Stack last 4 frame to create history
    env = FrameStack(env, k=4)
    # initialize the model 
    # image input so cnn 
    # convs = [n_outputs, karnel_size, stride]
    model = deepq.models.cnn_to_mlp(convs=[(32,3,1),(32,3,1)], hiddens=[256])
    # train the model
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-2,
        # number of iteration to optimizer for
        max_timesteps=10000,
        buffer_size=1000,
        # fraction of entire training period over which the exploration rate is annealed
        exploration_fraction=0.1,
        # final value of random action probability
        exploration_final_eps=0.01,
        print_freq=10
    )
    print("Saving model to breakout_model.pkl")
    act.save("breakout_model.pkl")


if __name__ == '__main__':
    main()
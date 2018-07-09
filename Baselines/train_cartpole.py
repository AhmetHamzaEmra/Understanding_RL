import gym

from baselines import deepq


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    # create the game enviroment
    env = gym.make("CartPole-v0")
    # initialize the model in this case 2 layered network 
    # with 64 hidden units
    model = deepq.models.mlp([64])
    # train the model
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        # number of iteration to optimizer for
        max_timesteps=130000,
        buffer_size=50000,
        # fraction of entire training period over which the exploration rate is annealed
        exploration_fraction=0.1,
        # final value of random action probability
        exploration_final_eps=0.02,
        # how often to print out training progress
        # set to None to disable printing
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
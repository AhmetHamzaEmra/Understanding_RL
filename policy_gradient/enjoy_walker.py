import gym
import os
import numpy as np
import tensorflow as tf
from itertools import product

env = gym.make('BipedalWalkerHardcore-v2')
obs = env.reset()

possible_torques = np.array([-1.0,-0.5, 0.0, 0.5, 1.0])
possible_actions = np.array(list(product(possible_torques, possible_torques, possible_torques, possible_torques)))




# 1. Specify the network architecture
n_inputs = env.observation_space.shape[0]  # == 24
n_hidden = 16
n_outputs = len(possible_actions) # == 625
initializer = tf.variance_scaling_initializer()

# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.selu, kernel_initializer=initializer)
hidden = tf.layers.dense(hidden, n_hidden, activation=tf.nn.selu, kernel_initializer=initializer)
hidden = tf.layers.dense(hidden, n_hidden, activation=tf.nn.selu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.softmax(logits)

action_index = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=-1)


saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./models/my_bipedal_walker_pg.ckpt")
    for i in range(5):
        obs = env.reset()
        game_score = 0
        while True:
            env.render()
            
            action_index_val = action_index.eval(feed_dict={X: obs.reshape(1, n_inputs)})

            action = possible_actions[action_index_val]
            obs, reward, done, info = env.step(action[0])

            game_score+=reward

            if done:
                break
        print("Game: {}, Score: {}".format(i+1, game_score) )


env.close()


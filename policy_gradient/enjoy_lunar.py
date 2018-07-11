import gym
import os
import numpy as np
import tensorflow as tf

env = gym.make('LunarLander-v2')
obs = env.reset()

n_inputs = 8
n_hidden = 32
n_outputs = 4

learning_rate = 0.001

initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
hidden = tf.layers.dense(hidden, n_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
hidden = tf.layers.dense(hidden, n_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.softmax(logits) 
action = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=-1)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./my_policy_net_pg.ckpt")
    for i in range(5):
        obs = env.reset()
        game_score = 0
        while True:
            env.render()
            
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})

            obs, reward, done, info = env.step(action_val[0])
            game_score+=reward

            if done:
                break
        print("Game: {}, Score: {}".format(i+1, game_score) )


env.close()
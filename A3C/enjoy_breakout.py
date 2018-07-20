import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import os
env = gym.make("LunarLander-v2")
obs = env.reset()

n_input = obs.shape[0]
n_actions = env.action_space.n


initializer = tf.variance_scaling_initializer()
# Create the neural network
def q_network(X_state, name):
    with tf.variable_scope(name) as scope :
        fc1 = tf.layers.dense(X_state, 16, activation=tf.nn.relu, kernel_initializer=initializer)
        outputs = tf.layers.dense(fc1, n_actions)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None,n_input])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")


init = tf.global_variables_initializer()
saver = tf.train.Saver()
checkpoint_path = "./models/my_dqn.ckpt"

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    score = 0 
    state = env.reset()
    for step in range(10000000):
        env.render()
        

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = np.argmax(q_values)
        # Online DQN plays
        state, reward, done, info = env.step(action)
        score+=reward

        if done:
            print(score)
            break









































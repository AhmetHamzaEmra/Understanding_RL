import numpy as np
import gym
import tensorflow as tf 
import matplotlib.pyplot as plt
from itertools import product

env = gym.make('BipedalWalker-v2')
possible_torques = np.array([-1.0, 0.0, 1.0])
possible_actions = np.array(list(product(possible_torques, possible_torques, possible_torques, possible_torques)))
possible_actions.shape

n_inputs = env.observation_space.shape[0]  # == 24
n_hidden = 64
n_outputs = len(possible_actions) # == 81

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(n_hidden, input_dim = n_inputs, activation='relu'))
model.add(tf.keras.layers.Dense(n_hidden, activation = "relu"))
model.add(tf.keras.layers.Dense(n_hidden, activation = "relu"))
model.add(tf.keras.layers.Dense(n_outputs, activation = "softmax"))
model.build()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.summary()

def discount_rewards(r, gamma = 0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

episodes = 5000
scores = []
update_every = 5

gradBuffer = model.trainable_variables
for ix,grad in enumerate(gradBuffer):
  gradBuffer[ix] = grad * 0
  
for e in range(episodes):
  
  s = env.reset()
  
  ep_memory = []
  ep_score = 0
  done = False 
  while not done: 
    s = s.reshape([1,24])
    with tf.GradientTape() as tape:
      #forward pass
      logits = model(s)
      a_dist = logits.numpy()
      # Choose random action with p = action dist
      a = np.random.choice(a_dist[0],p=a_dist[0])
      a = np.argmax(a_dist == a)
      loss = compute_loss([a], logits)
    # make the choosen action 
    a = possible_actions[a]
    s, r, done, _ = env.step(a)
    ep_score +=r
    grads = tape.gradient(loss, model.trainable_variables)
    ep_memory.append([grads,r])
  scores.append(ep_score)
  # Discound the rewards 
  ep_memory = np.array(ep_memory)
  ep_memory[:,1] = discount_rewards(ep_memory[:,1])
  
  for grads, r in ep_memory:
    for ix,grad in enumerate(grads):
      gradBuffer[ix] += grad * r
  
  if e % update_every == 0:
    optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
    for ix,grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0
      
  if e % 50 == 0:
    print("Episode  {}  Score  {}".format(e, np.mean(scores[-100:])))


# coding: utf-8

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.initializers import uniform
import keras.backend as K

from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork

state_size = 8
action_size = 3
action_bound = 3.
batch_size = 16
mixing_rate = 0.8
learning_rate = 0.001

state = np.random.random([batch_size, state_size])
action = np.random.random([batch_size, action_size])
action_grad = np.random.random([batch_size, action_size])

with tf.Session() as sess:
    actor = ActorNetwork(sess, state_size, action_size, action_bound, batch_size, mixing_rate, learning_rate)
    actor.train(state, action_grad)

    critic = CriticNetwork(sess, state_size, action_size, batch_size, mixing_rate, learning_rate)
    target_q = actor.predict_target(state)
    critic.train(state, action, target_q)

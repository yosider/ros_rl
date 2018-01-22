# coding: utf-8

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Lambda, BatchNormalization
from keras.layers.merge import concatenate
from keras.initializers import uniform
from keras.optimizers import Adam
import keras.backend as K

from constants import *

# for BN layer
K.set_learning_phase(1)


class CriticNetwork(object):
    """
    input: state, action
    output: Q(state, action)
    """
    def __init__(self, sess, state_size, action_size, minibatch_size, mixing_rate, learning_rate):
        self.sess = sess
        self.minibatch_size = minibatch_size
        self.tau = mixing_rate
        self.learning_rate = learning_rate

        K.set_session(sess)

        self.model, self.state, self.action = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_state, self.target_action = self.create_critic_network(state_size, action_size)

#        # target Q value (given by target critic net.)
#        self.target_q = tf.placeholder(tf.float32, [None, 1])
#        # critic's loss (TD error)
#        self.loss = K.mean_square_error(self.target_q, self.model.output)
#        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # dQ/da (pass to actor)
        self.action_grads = tf.gradients(self.model.output, self.action)

        self.sess.run(tf.global_variables_initializer())

    def train(self, state, action, target_q):
        # input shape: [(BATCH_SIZE, STATE_SIZE), (BATCH_SIZE, ACTION_SIZE)]
        # TODO: can return loss value.
        self.model.fit([state, action], target_q, batch_size=self.minibatch_size, epochs=2)

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1-self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def predict(self, state, action):
        return self.model.predict([state, action])

    def predict_target(self, state, action):
        return self.target_model.predict([state, action])

    def action_gradients(self, state, action):
        """ return dQ/da """
        return self.sess.run(self.action_grads, feed_dict={
            self.state: state,
            self.action: action,
        })[0]

    def create_critic_network(self, state_size, action_size):
        state = Input(shape=[state_size])  # placeholder
        action = Input(shape=[action_size])
        h = Dense(64, kernel_initializer='he_uniform', activation='relu')(state)
        h = BatchNormalization()(h)
        h = concatenate([h, action])
        h = Dense(32, kernel_initializer='he_uniform', activation='relu')(h)
        qval = Dense(action_size, kernel_initializer='random_uniform', activation='linear')(h)
        model = Model(inputs=[state,action], outputs=qval)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model, state, action

# coding: utf-8

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Lambda, BatchNormalization
from keras.initializers import uniform
import keras.backend as K

from constants import *


class ActorNetwork(object):
    """
    input: state
    output: action
    """
    def __init__(self, sess, state_size, action_size, action_bound, minibatch_size, mixing_rate, learning_rate):
        self.sess = sess
        self.tau = mixing_rate
        self.learning_rate = learning_rate

        K.set_session(sess)

        self.model, self.params, self.state = self.create_actor_network(state_size, action_size, action_bound)
        self.target_model, self.target_params, self.target_state = self.create_actor_network(state_size, action_size, action_bound)

        # dQ/da
        self.action_grad = tf.placeholder(tf.float32, [None, action_size])
        # dQ/dθ = dQ/da * da/dθ
        params_grad = tf.gradients(self.model.output, self.params, -self.action_grad)
        # normalization
        self.params_grad = [x/minibatch_size for x in params_grad]
        #self.params_grad = list(map(lambda x: tf.div(x, minibatch_size), params_grad))
        # (grad, param) pairs
        grads = zip(self.params_grad, self.params)
        # optimizer
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)

        self.sess.run(tf.global_variables_initializer())

    def train(self, state, action_grad):
        # input shape: [(BATCH_SIZE, STATE_SIZE), (BATCH_SIZE, ACTION_SIZE)]
        for i in range(0, BATCH_SIZE-MINIBATCH_SIZE, MINIBATCH_SIZE):
            self.sess.run(self.optimize, feed_dict={
                self.state: state[i:i+MINIBATCH_SIZE],
                self.action_grad: action_grad[i:i+MINIBATCH_SIZE]
                })

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1-self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def predict(self, state):
        return self.model.predict(state)

    def predict_target(self, state):
        return self.target_model.predict(state)

    def create_actor_network(self, state_size, action_size, action_bound):
        state = Input(shape=[state_size])  # placeholder
        h = Dense(64, kernel_initializer='he_uniform', activation='relu')(state)
        h = BatchNormalization()(h)
        h = Dense(32, kernel_initializer='he_uniform', activation='relu')(h)
        action = Dense(action_size, kernel_initializer='random_uniform', activation='tanh')(h)
        action = Lambda(lambda x: x * action_bound)(action)
        model = Model(inputs=state, outputs=action)

        return model, model.trainable_weights, state

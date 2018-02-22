# coding: utf-8

import numpy as np

import tensorflow as tf
import tflearn

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

        self.output, self.state = self.create_actor_network(state_size, action_size, action_bound)
        self.params = tf.trainable_variables()
        ### WIP ###
        self.target_output, self.target_state = self.create_actor_network(state_size, action_size, action_bound)
        self.target_params = tf.trainable_variables()[len(self.params):]

        # dQ/da
        self.action_grad = tf.placeholder(tf.float32, [None, action_size])
        # dQ/dθ = dQ/da * da/dθ
        params_grad = tf.gradients(self.output, self.params, -self.action_grad)
        # normalization
        self.params_grad = [g / float(minibatch_size) for g in params_grad]
        # (grad, param) pairs
        grads = zip(self.params_grad, self.params)
        # optimizer
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)

        #TODO:
        #self.sess.run(tf.global_variables_initializer())

    def train(self, state, action_grad):
        # input shape: [(BATCH_SIZE, STATE_SIZE), (BATCH_SIZE, ACTION_SIZE)]
        #indices = np.arange(BATCH_SIZE)
        for epoch in range(ACTOR_EPOCHS):
            for i in range(MINIBATCH_SIZE, BATCH_SIZE, MINIBATCH_SIZE):
                self.sess.run(self.optimize, feed_dict={
                    self.state: state[i-MINIBATCH_SIZE : i],
                    self.action_grad: action_grad[i-MINIBATCH_SIZE : i]
                    })
            #np.random.shuffle(indices)
            #state = state[indices]
            #action_grad = action_grad[indices]
            
    def update_target(self):
        [self.target_params[i].assign( \
            tf.multiply(self.params[i], self.tau) + \
            tf.multiply(self.target_params[i], 1.-self.tau) \
            ) for i in range(len(self.target_params))]

    def predict(self, state):
        return self.sess.run(self.output, feed_dict={
            self.state: state
        })

    def predict_target(self, state):
        return self.sess.run(self.target_output, feed_dict={
            self.target_state: state
        })

    def get_num_trainable_vars(self):
        return len(self.params) + len(self.target_params)

    def create_actor_network(self, state_size, action_size, action_bound):
        state = tflearn.input_data(shape=[None, state_size])
        net = tflearn.fully_connected(state, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        output = tflearn.fully_connected(
            net, action_size, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        output = tf.multiply(output, action_bound)
        
        return output, state



class CriticNetwork(object):
    """
    input: state, action
    output: Q(state, action)
    """
    def __init__(self, sess, state_size, action_size, minibatch_size, mixing_rate, learning_rate, num_actor_vars):
        self.sess = sess
        self.minibatch_size = minibatch_size
        self.tau = mixing_rate
        self.learning_rate = learning_rate

        self.output, self.state, self.action = self.create_critic_network(state_size, action_size)
        self.params = tf.trainable_variables()[num_actor_vars:]
        self.target_output, self.target_state, self.target_action = self.create_critic_network(state_size, action_size)
        self.target_params = tf.trainable_variables()[(num_actor_vars + len(self.params)):]

        # target Q value (given by target critic net.)
        self.target_q = tf.placeholder(tf.float32, [None, 1])
        # critic's loss (TD error)
        self.loss = tflearn.mean_square(self.target_q, self.output)
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # dQ/da (pass to actor)
        self.action_grads = tf.gradients(self.output, self.action)

        #TODO:
        #self.sess.run(tf.global_variables_initializer()) 

    def train(self, state, action, target_q):
        # input shape: [(BATCH_SIZE, STATE_SIZE), (BATCH_SIZE, ACTION_SIZE)]
        #indices = np.arange(BATCH_SIZE)
        #TODO
        #TODO self.out
        #TODO
        for epoch in range(CRITIC_EPOCHS):
            for i in range(MINIBATCH_SIZE, BATCH_SIZE, MINIBATCH_SIZE):
                self.sess.run(self.optimize, feed_dict={
                    self.state: state[i-MINIBATCH_SIZE : i],
                    self.action: action[i-MINIBATCH_SIZE : i],
                    self.target_q: target_q[i-MINIBATCH_SIZE : i],
                    })
            #np.random.shuffle(indices)
            #state = state[indices]
            #action_grad = action_grad[indices]

    def update_target(self):
        [self.target_params[i].assign( \
            tf.multiply(self.params[i], self.tau) + \
            tf.multiply(self.target_params[i], 1-self.tau) \
            ) for i in range(len(self.target_params))]

    def predict(self, state, action):
        return self.sess.run(self.output, feed_dict={
            self.state: state,
            self.action: action
        })

    def predict_target(self, state, action):
        return self.sess.run(self.target_output, feed_dict={
            self.target_state: state,
            self.target_action: action
        })

    def action_gradients(self, state, action):
        """ return dQ/da """
        return self.sess.run(self.action_grads, feed_dict={
            self.state: state,
            self.action: action,
        })[0]

    def create_critic_network(self, state_size, action_size):
        state = tflearn.input_data(shape=[None, state_size])
        action = tflearn.input_data(shape=[None, action_size])
        net = tflearn.fully_connected(state, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)
        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        output = tflearn.fully_connected(net, 1, weights_init=w_init)
        
        return output, state, action
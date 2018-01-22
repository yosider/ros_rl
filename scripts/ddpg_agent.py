#!/usr/bin/env python
# coding: utf-8

import rospy
from rospy.numpy_msg import numpy_msg
from ros_rl.msg import Stepinfo, Floats

import numpy as np
import tensorflow as tf

from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork
from networks.buffers import ReplayBuffer
from networks.constants import *
from utils import *

class DDPG_agent():
    def __init__(self):
        self.sub_stepinfo = rospy.Subscriber("/ros_rl/stepinfo", Stepinfo, self.step, queue_size=1, buff_size=2**24)
        #self.sub_batch = rospy.Subscriber("/ros_rl/batch", numpy_msg(Batch), self.train, queue_size=1, buff_size=2**24)
        self.pub_action = rospy.Publisher("/ros_rl/action", numpy_msg(Floats), queue_size=1)

        self.sess = tf.InteractiveSession()
        self.actor = ActorNetwork(self.sess, STATE_SIZE, ACTION_SIZE, ACTION_BOUND, MINIBATCH_SIZE, TAU, ACTOR_LEARNING_RATE)
        self.critic = CriticNetwork(self.sess, STATE_SIZE, ACTION_SIZE, MINIBATCH_SIZE, TAU, CRITIC_LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.graph = tf.get_default_graph()

        self.state = np.zeros([1, STATE_SIZE])
        self.action = np.zeros(ACTION_SIZE)
        self.count = 0

    def step(self, msg):
        next_state, reward, terminal, info = unpack_stepinfo(msg)
        #assert next_state.shape == (1, STATE_SIZE)
        #TODO: prioritize
        self.replay_buffer.add(self.state[0], self.action, reward, terminal, next_state[0])
        self.count += 1

        if self.count >= BATCH_SIZE:
            self.train()
            self.count = 0

        with self.graph.as_default():
            action = self.actor.predict(next_state)[0]
            #assert action.shape == (ACTION_SIZE, )
            print "action:", action
            self.action = action
            self.state = next_state
            self.pub_action.publish(action)

    def train(self):
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(BATCH_SIZE)

        with self.graph.as_default():
            # target Q value
            target_action = self.actor.predict_target(s2_batch)
            target_Q = self.critic.predict_target(s2_batch, target_action)

            # target R value
            target_R = []
            for i in range(BATCH_SIZE):
                if t_batch[i]:
                    # terminal state
                    target_R.append(r_batch[i])
            	else:
	                target_R.append(r_batch[i] + GAMMA * target_Q[i])
            target_R = np.array(target_R).reshape(BATCH_SIZE, 1)

            # train critic
            self.critic.train(s_batch, a_batch, target_R)

            pred_action = self.actor.predict(s_batch)
            dqda = self.critic.action_gradients(s_batch, pred_action)
            self.actor.train(s_batch, dqda)

            # Update target networks
            self.actor.update_target()
            self.critic.update_target()


if __name__ == '__main__':
    DDPG_agent()
    rospy.init_node('ddpg_agent', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down \"ddpg_agent\" node...")

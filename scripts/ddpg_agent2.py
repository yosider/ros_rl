#!/usr/bin/env python
# coding: utf-8
from datetime import datetime 

import rospy
from rospy.numpy_msg import numpy_msg
from ros_rl.msg import Stepinfo, Floats

import numpy as np
import tensorflow as tf

from networks.networks2 import ActorNetwork
from networks.networks2 import CriticNetwork
from networks.buffers import ReplayBuffer
from networks.constants import *
from utils import *

LOGDIR = '/home/yosider/robo_ws/src/ros_rl/logs/' + ENV_NAME + '/' + str(datetime.now()) + '_CKPT_test2' + '/'
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

class DDPG_agent():
    def __init__(self):
        self.sub_stepinfo = rospy.Subscriber("/ros_rl/stepinfo", Stepinfo, self.step, queue_size=1, buff_size=2**24)
        #self.sub_batch = rospy.Subscriber("/ros_rl/batch", numpy_msg(Batch), self.train, queue_size=1, buff_size=2**24)
        self.pub_action = rospy.Publisher("/ros_rl/action", numpy_msg(Floats), queue_size=1)

        self.sess = tf.InteractiveSession()
        self.actor = ActorNetwork(self.sess, STATE_SIZE, ACTION_SIZE, ACTION_BOUND, MINIBATCH_SIZE, TAU, ACTOR_LEARNING_RATE)
        self.critic = CriticNetwork(self.sess, STATE_SIZE, ACTION_SIZE, MINIBATCH_SIZE, TAU, CRITIC_LEARNING_RATE, self.actor.get_num_trainable_vars())
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.graph = tf.get_default_graph()
        self.sess.run(tf.global_variables_initializer())
        # TODO: confirm

        self.state = np.zeros([1, STATE_SIZE])
        self.action = np.zeros(ACTION_SIZE)
        self.count = 0
        self.train_times = 0

    def step(self, msg):
        next_state, reward, terminal, info = unpack_stepinfo(msg)
        #assert next_state.shape == (1, STATE_SIZE)
        #TODO: prioritize
        self.replay_buffer.add(self.state[0], self.action, reward, terminal, next_state[0])
        self.count += 1

        if self.count >= BATCH_SIZE:
            self.train()
            #self.count = 0
            self.train_times += 1

        with self.graph.as_default():
            action = self.actor.predict(next_state)[0]
            #assert action.shape == (ACTION_SIZE, )
            #print "action:", action
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
    agent = DDPG_agent()
    rospy.init_node('ddpg_agent', anonymous=True)
    rospy.spin()
    
    #print("Saving models...")
    #agent.actor.model.save(LOGDIR + '/actor-batch' + str(agent.train_times) + '.tflearn')
    #agent.critic.model.save(LOGDIR + '/critic-batch' + str(agent.train_times) + '.tflearn')
    print("Shutting down the node...")

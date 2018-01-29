#!/usr/bin/env python
# coding: utf-8

import rospy
#TODO : try numpy_msg(Stepinfo)
from std_msgs.msg import Float32
from rospy.numpy_msg import numpy_msg
from ros_rl.msg import Stepinfo, Floats

import numpy as np
import gym

from matplotlib import pyplot as plt 

from utils import *
from networks.constants import *

# Log
LOGDIR = '/home/yosider/robo_ws/src/ros_rl/logs/' + ENV_NAME + '/' + str(time.time()) + '/'
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

class Env():
    def __init__(self):
        self.env = ENV

        self.sub_action = rospy.Subscriber("/ros_rl/action", numpy_msg(Floats), self.step, queue_size=1, buff_size=2**24)
        self.pub_stepinfo = rospy.Publisher("/ros_rl/stepinfo", Stepinfo, queue_size=1)
        self.pub_ep_reward = rospy.Publisher("/ros_rl/stats/ep_reward", Float32, queue_size=4)
        rospy.init_node('env', anonymous=True)

        self.log_reward = []
        self.ep_num = 0
        self.ep_step = 0
        self.ep_reward = 0
        #self.log = open('log_reward.txt', 'w')

        self.reset()
        rospy.loginfo('initialization done.')

    def reset(self):
        self.ep_num += 1
        s, r, t, i = self.env.reset(), 0, False, ''
        stepinfo = pack_stepinfo(s, r, t, i)
        rospy.loginfo('Episode: %d', self.ep_num)
        #print 'ep:', self.ep_num, 'state:', s
        self.pub_stepinfo.publish(stepinfo)

    def step(self, msg):
        action = msg.data
        # shape: (self.env.action_space.shape, )

        s, r, t, i = self.env.step(action)

        self.ep_step += 1
        self.ep_reward += r
        if t==1 or self.ep_step==MAX_EP_STEPS:
            rospy.loginfo('Finish at step %d, reward %d', self.ep_step, int(self.ep_reward))
            self.log_reward.append(self.ep_reward)
            self.ep_step = 0
            self.ep_reward = 0
            self.reset()
        else:
            stepinfo = pack_stepinfo(s, r, t, i)
            self.pub_stepinfo.publish(stepinfo)

    def visualize_log(self):
        plt.plot(self.log_reward)
        plt.savefig(LOGDIR + 'rewards.png')


if __name__ == '__main__':
    env = Env()
    rospy.spin()
    env.visualize_log()
#!/usr/bin/env python
# coding: utf-8

import rospy
#TODO : try numpy_msg(Stepinfo)
from rospy.numpy_msg import numpy_msg
from ros_rl.msg import Stepinfo, Floats

import numpy as np
import gym

from utils import *

class Env():
    def __init__(self):
        self.env = gym.make('Pendulum-v0')

        self.sub = rospy.Subscriber("/ros_rl/action", numpy_msg(Floats), self.step, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher("/ros_rl/stepinfo", Stepinfo, queue_size=1)
        rospy.init_node('env', anonymous=True)

        self.reset()
        rospy.loginfo('initialization done.')

    def reset(self):
        s, r, t, i = self.env.reset(), 0, False, ''
        stepinfo = pack_stepinfo(s, r, t, i)
        rospy.loginfo('Reset Environment.')
        print 'state:', s
        self.pub.publish(stepinfo)

    def step(self, msg):
        action = msg.data
        assert action.shape == self.env.action_space.shape
        s, r, t, i = self.env.step(action)
        stepinfo = pack_stepinfo(s, r, t, i)
        print 'state:', s
        self.pub.publish(stepinfo)


if __name__ == '__main__':
    Env()
    rospy.spin()

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gym

import cv2
from cv_bridge import CvBridge
from ros_rl.msg import Stepinfo, Batch

from networks.constants import *

def pack_stepinfo(s, r, t, i):
    # TODO: CvBridge() for 2dim.
    # TODO: type(info)==dictでした．
    stepinfo = Stepinfo()
    stepinfo.state = list(s)
    stepinfo.reward = r
    if type(t) == bool:
        t = int(t)
    stepinfo.terminal = t
    return stepinfo

def unpack_stepinfo(stepinfo):
    s = np.array(stepinfo.state).reshape(1,-1)
    r = stepinfo.reward
    t = stepinfo.terminal
    i = stepinfo.info
    return s, r, t, i

def pack_batch(s, a, r, t, s2):
    # s.shape: (BATCH_SIZE, STATE_SIZE) for 1-D.
    batch = Batch()
    batch.state = CvBridge().cv2_to_imgmsg(s)
    batch.action = CvBridge().cv2_to_imgmsg(a)
    batch.reward = list(r)
    batch.terminal = list(t)
    batch.next_state = CvBridge().cv2_to_imgmsg(s2)
    return batch

def unpack_batch(batch):
    s = CvBridge().imgmsg_to_cv2(batch.state)
    a = CvBridge().imgmsg_to_cv2(batch.action)
    r = np.array(batch.r)
    t = np.array(batch.t)
    s2 = CvBridge().imgmsg_to_cv2(batch.next_state)
    return s, a, r, t, s2

def render_result(actor, critic):
    env = gym.make(ENV_NAME)
    for ep_num in range(5):
        s, t = env.reset(), False
        ep_reward = 0
        ep_step = 0
        while not t:
            env.render()
            action = actor.predict(np.array(s).reshape(1,-1))[0]
            s, r, t, i = env.step(action)
            ep_reward += r
        print('Episode %d finished at step %d, reward %d.' % (ep_num, ep_step, ep_reward))



# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

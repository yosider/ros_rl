# coding: utf-8
"""
DDPG
http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
"""

import os
import time
import numpy as np
import gym
import tensorflow as tf

from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork
from networks.buffers import ReplayBuffer
#from utils import OrnsteinUhlenbeckActionNoise

ENV = gym.make('Pendulum-v0')

STATE_SIZE = ENV.observation_space.shape[0]
ACTION_SIZE = ENV.action_space.shape[0]
ACTION_BOUND = ENV.action_space.high
# Ensure action bound is symmetric
assert (ENV.action_space.high == -ENV.action_space.low)

# mixing rate for target network update
TAU = 0.001
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 64
TRAIN_INTERVAL = 1
MAX_EPISODES = 200
MAX_EP_STEPS = 300
GAMMA = 0.99
LOGDIR = 'logs/' + '/nonros/' + 'h64s/' #str(int(time.time())) + '/'
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
#else:
#    print('LOGDIR already exists. You are (un)lucky :)')
#    exit(1)

# log variables
reward_log = []
q_log = []

def main():
    with tf.Session() as sess:

        actor = ActorNetwork(sess, STATE_SIZE, ACTION_SIZE, ACTION_BOUND, MINIBATCH_SIZE, TAU, ACTOR_LEARNING_RATE)
        critic = CriticNetwork(sess, STATE_SIZE, ACTION_SIZE, MINIBATCH_SIZE, TAU, CRITIC_LEARNING_RATE)

        #actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ACTION_SIZE))

        #TODO: Ornstein-Uhlenbeck noise.

        sess.run(tf.global_variables_initializer())

        # initialize target net
        actor.update_target()
        critic.update_target()

        # initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE)

        #graph = tf.get_default_graph()

        # main loop.
        for ep in range(MAX_EPISODES):

            episode_reward = 0
            ep_batch_avg_q = 0

            s = ENV.reset()

            for step in range(MAX_EP_STEPS):
                #with graph.as_default():
                a = actor.predict(np.reshape(s, (1, STATE_SIZE)))
                s2, r, terminal, info = ENV.step(a[0])
                #print(s2)

                replay_buffer.add(np.reshape(s, (STATE_SIZE,)), \
                                np.reshape(a, (ACTION_SIZE,)), \
                                r, \
                                terminal, \
                                np.reshape(s2, (STATE_SIZE,)))

                # Batch sampling.
                if replay_buffer.size() > MINIBATCH_SIZE and \
                    step % TRAIN_INTERVAL == 0:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # target Q値を計算．
                    target_action = actor.predict_target(s2_batch)
                    target_q = critic.predict_target(s2_batch, target_action)

                    # critic の target V値を計算．
                    targets = []
                    for i in range(MINIBATCH_SIZE):
                        if t_batch[i]:
                            # terminal
                            targets.append(r_batch[i])
                        else:
                            targets.append(r_batch[i] + GAMMA * target_q[i])

                    # Critic を train.
                    #TODO: predQはepisodeではなくrandom batchなのでepisode_avg_maxという統計は不適切．
                    critic.train(s_batch, a_batch, np.reshape(targets, (MINIBATCH_SIZE, 1)))

                    # Actor を　train.
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    #print(grads.shape)
                    #exit(1)
                    actor.train(s_batch, grads)

                    # Update target networks.
                    # 数batchに一度にするべき？
                    actor.update_target()
                    critic.update_target()

                    #ep_batch_avg_q += np.mean(pred_q)

                s = s2
                episode_reward += r

                if terminal:
                    print('Episode:', ep, 'Reward:', episode_reward)
                    reward_log.append(episode_reward)
                    q_log.append(ep_batch_avg_q / step)

                    break


import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def visualize(log, x_name, y_name):
    plt.plot(log)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(LOGDIR + y_name +'.png')
    plt.clf()


if __name__ == '__main__':
    try:
        main()
        visualize(reward_log, 'episode', 'episode-reward')
        #visualize(q_log, 'episode', 'batch-avg-Q')
    except KeyboardInterrupt:
        visualize(reward_log, 'episode', 'episode-reward')
        #visualize(q_log, 'episode', 'batch-avg-Q')

# coding: utf-8
import gym
import os
from datetime import datetime


# --- FLAGS
LOGGING = True
TRAINING = True
RENDERING = False
LOAD_MODELS = False
RENDER_AFTER_TRAIN = True

# --- Environment
ENV_NAME = 'Pendulum-v0'
ENV = gym.make(ENV_NAME)
STATE_SIZE = ENV.observation_space.shape[0]
ACTION_SIZE = ENV.action_space.shape[0]
ACTION_BOUND = ENV.action_space.high

# --- Logfile Directory
LOG_ROOT = '/home/yosider/robo_ws/src/ros_rl/logs/' + ENV_NAME + '/'
LOGDIR = LOG_ROOT + datetime.now().isoformat()[:16] + '/'
if LOGGING and not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

# --- Model Loading
LOAD_DIR = LOG_ROOT + '2018-02-11 13:25:44.221392_CKPT/'
ACTOR_FILE = LOAD_DIR + 'actor-batch112508.h5'
#ACTOR_TARGET_FILE = LOAD_DIR + ''
CRITIC_FILE = LOAD_DIR + 'critic-batch112508.h5'
#

# --- Training (Network)
TAU = 1e-3
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
GAMMA = 0.99

# --- Training (Data)
# size of data used in a single update
MINIBATCH_SIZE = 64
# size of data used in a single train (1 train means 32 times update)
BATCH_SIZE = 64
CRITIC_EPOCHS = 1
ACTOR_EPOCHS = 1
BUFFER_SIZE = 100000
MAX_EPISODES = 200
MAX_EP_STEPS = 300

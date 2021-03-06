# coding: utf-8
import gym

# --- Environment
ENV = gym.make('Pendulum-v0')
STATE_SIZE = ENV.observation_space.shape[0]
ACTION_SIZE = ENV.action_space.shape[0]
ACTION_BOUND = ENV.action_space.high

# --- Training (Network)
TAU = 1e-3
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
GAMMA = 0.99

# --- Training (Data)
# size of data used in a single update
MINIBATCH_SIZE = 64
# size of data used in a single train (1 train means 32 times update)
BATCH_SIZE = 320
CRITIC_EPOCHS = 1
ACTOR_EPOCHS = 1
BUFFER_SIZE = 100000
MAX_EPISODES = 200
MAX_EP_STEPS = 1000

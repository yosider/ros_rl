# coding: utf-8
import numpy as np

import rospy
from std_msgs import Int8
from raspirobo.msg import Floats
from ros_rl.msg import Stepinfo, Batch

from networks.buffers import ReplayBuffer
from constants import *
from utils import *

class Memory():
    def __init__(self):
        self.sub_q = rospy.Subscriber("/ros_rl/q_value", Floats, self.recall, queue_size=1, buffer_size=2**24)
        self.pub_recall = rospy.Publisher("/ros_rl/recall", Stepinfo, queue_size=1)

        self.pub_learning_signal = rospy.Publisher("/ros_rl/learning_signal", Int8, queue_size=1)
        self.sub_learning_

        self.sub_stepinfo = rospy.Subscriber("/ros_rl/stepinfo", Stepinfo, self.memorize, queue_size=1, buffer_size=2**24)

        self.pub_batch = rospy.Publisher("/ros_rl/batch", Batch, queue_size=1)

        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def recall(self):
        pass

    def batch_cb(self):
        batch = self.buffer.sample_batch(BATCH_SIZE)
        msg = pack_batch(*batch)
        self.pub_batch.publish(msg)

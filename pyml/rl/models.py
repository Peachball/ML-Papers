import sys
sys.path.append("..")

import gym
from utils import *
import time
import numpy as np
import matplotlib.pyplot as plt


class ICMModule():
    def __init__(self, inp, act):
        self.input = inp
        self.action = act
        with tf.variable_scope("forward"):
            net = tf.concat([inp[:-1], act[:-1]], 1)
            net = tf.layers.dense(net, 40, name="fc1")
            net = tf.nn.elu(net)
            net = tf.layers.dense(net, 40, name="fc2")
            net = tf.nn.elu(net)
            net = tf.layers.dense(net, inp.get_shape().as_list()[1], name="fc3")
            pred_state = net
            tf.summary.histogram("icm_predicted", pred_state, collections=['icm'])
            tf.summary.histogram("icm_actual", inp[1:], collections=['icm'])
            self.r = tf.norm(pred_state - inp[1:], axis=1)
        with tf.variable_scope("inverse"):
            net = tf.concat([inp[:-1], inp[1:]], 1)
            net = tf.layers.dense(net, act.get_shape().as_list()[1], name="fc")
            pred_act = net
        self.forward_loss = tf.reduce_sum(tf.square(pred_state - inp[1:]),
                axis=1)
        self.inverse_loss = tf.losses.softmax_cross_entropy(
                act[:-1], pred_act)


class DiscreteModel():
    def __init__(self, inp, act_space):
        self.input = inp
        assert isinstance(act_space, gym.spaces.Discrete)
        self.action_space = act_space
        self.policy, self.value = self.build_net(self.input)
        self.sample = tf.squeeze(tf.multinomial(self.policy, 1))

    def build_net(self, inp):
        net = inp
        policy = tf.layers.dense(net, self.action_space.n,
                kernel_initializer=tf.random_normal_initializer(stddev=0.001), name="policy")
        value = tf.layers.dense(net, 1, name="value",
                kernel_initializer=tf.random_normal_initializer(stddev=0.1), use_bias=False)
        return policy, value


class BoxModel():
    def __init__(self, inp, act_space):
        self.input = inp
        assert isinstance(act_space, gym.spaces.Box)
        self.action_space = act_space

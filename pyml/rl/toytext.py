#!/usr/bin/env python3
from gridworld import GridWorld
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
            net = tf.layers.dense(net, inp.get_shape().as_list()[1], name="fc")
            pred_state = net
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
                kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="policy")
        value = tf.layers.dense(net, 1, name="value", use_bias=False)
        return policy, value


def lakerunner(LOGDIR="tflogs/gridworld", ENV="gridworld"):
    if ENV == 'gridworld':
        env = GridWorld()
    else:
        env = gym.make(ENV)
    X = tf.placeholder(tf.int32, [None], name="state")
    R = tf.placeholder(tf.float32, [None], name="R")
    A = tf.placeholder(tf.int32, [None], name="action")
    exp_x = tf.one_hot(X, env.observation_space.n)
    expanded_a = tf.one_hot(A, env.action_space.n)
    with tf.variable_scope("model"):
        m = DiscreteModel(exp_x, env.action_space)
        icm = ICMModule(exp_x, expanded_a)
    m.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")

    log_policy = tf.nn.log_softmax(m.policy)
    R_e = tf.expand_dims(R, axis=1)
    value_loss = tf.reduce_sum(tf.square(m.value - R_e))
    advantage = tf.stop_gradient(R_e - m.value)
    policy_loss = -tf.reduce_sum(expanded_a * log_policy * advantage)
    entropy = - tf.reduce_sum(tf.nn.softmax(m.policy) * log_policy)
    bs = tf.cast(tf.shape(X)[0], tf.float32)
    tf.summary.scalar("model/entropy", entropy / bs)
    tf.summary.scalar("model/policy_loss", policy_loss / bs)
    tf.summary.scalar("model/value_loss", value_loss / bs)

    forward_loss = tf.reduce_sum(icm.forward_loss)
    inverse_loss = tf.reduce_sum(icm.inverse_loss)
    tf.summary.scalar("icm/forward_loss", forward_loss / bs)
    # tf.summary.scalar("icm/inverse_loss", inverse_loss / bs)

    beta = 0.9
    loss = 0.25 * value_loss + policy_loss - 0.01 * entropy
    icm_loss = beta * forward_loss + (1 - beta) * inverse_loss

    summary_op = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss,
            global_step=global_step)
    icm_train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(icm_loss,
            global_step=global_step)

    sw = tf.summary.FileWriter(LOGDIR)
    sv = tf.train.Supervisor(logdir=LOGDIR,
            summary_writer=None,
            summary_op=None,
            save_model_secs=30)

    plt.ion()
    gpu_config = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_config)
    with sv.managed_session(config=config) as sess:
        sw.add_graph(sess.graph)
        iters = 0
        while True:
            done = False
            obs = []
            rew = []
            total_reward = []
            total_states = []
            act = []

            o = env.reset()
            def train(last_o, terminal=False, gamma=0.9, curiosity=False,
                    mpl=True):
                if len(obs) == 0:
                    return
                estimated_R = []
                ir = sess.run(icm.r, feed_dict={X: obs + [last_o], A: act + [0]})
                if not terminal:
                    rR = sess.run(m.value, feed_dict={X: [last_o]})[0, 0]
                else:
                    rR = 0

                for mr, inr in zip(rew[::-1], ir[::-1]):
                    if curiosity:
                        rR = gamma * rR + inr + mr
                    else:
                        rR = gamma * rR + mr
                    estimated_R = [rR] + estimated_R
                s, _ = sess.run([summary_op, train_op],
                        feed_dict={X: obs, A: act, R: estimated_R})
                sw.add_summary(s, sess.run(global_step))
                if curiosity:
                    sess.run(icm_train_op,
                            feed_dict={X: obs + [last_o],
                                A: act + [0] # Note the [0] is a filler
                                })
                nonlocal iters
                if mpl and iters % 100 == 0:
                    print("Updating image")
                    states = np.arange(16)
                    v = sess.run(m.value,
                            feed_dict={X: states})
                    disp = np.zeros((6, 6))
                    disp[1:5, 1:5] = v.reshape((4, 4))
                    disp[5, 5] = 1
                    plt.imshow(disp, cmap='hot', interpolation='none')
                    plt.pause(0.01)
                iters += 1
            visual = False
            while not done:
                obs.append(o)
                total_states.append(o)
                if visual:
                    env.render()
                    time.sleep(0.1)
                a, a_dist = sess.run([m.sample, m.policy], feed_dict={X: [o]})
                np.set_printoptions(suppress=True)
                new_o, r, done, i = env.step(a)
                total_reward += [r]

                act.append(a)
                rew.append(r)

                if len(obs) > 32:
                    train(new_o)
                    obs = []
                    rew = []
                    act = []
                o = new_o

            train(new_o, terminal=True)
            s = tf.Summary(value=[
                tf.Summary.Value(tag="reward", simple_value=sum(total_reward)),
                tf.Summary.Value(tag="length", simple_value=len(total_reward))
                ])
            sw.add_summary(s, global_step=sess.run(global_step))


def play_text_env(ENV='gridworld'):
    if ENV == 'gridworld':
        env = GridWorld()
    else:
        env = gym.make(ENV)
    move_to_num = {'l': 0, 'd': 1, 'r': 2, 'u': 3}
    env.reset()
    while True:
        env.render()
        m = input().strip()
        _, r, d, _ = env.step(move_to_num[m])
        print("Reward: ", r)
        if d:
            env.render()
            print("Game over")
            env.reset()


def play_taxi_env(ENV='Taxi-v2'):
    env = gym.make(ENV)
    s_to_a = {'d': 0, 'u': 1, 'r': 2, 'l': 3, 'p': 4, 'o': 5}
    env.reset()
    while True:
        env.render()
        m = input().strip()
        _, r, d, _ = env.step(s_to_a[m])
        print("Reward: ", r)
        if d:
            env.render()
            print("Game over")
            env.reset()


if __name__ == '__main__':
    lakerunner()

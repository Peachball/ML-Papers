#!/usr/bin/env python3
import sys
sys.path.append("..")

import gym
from utils import *
import time
import numpy as np
import matplotlib.pyplot as plt


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


def lakerunner(LOGDIR="tflogs/frozenlake", ENV="FrozenLake-v0"):
    env = gym.make(ENV)
    X = tf.placeholder(tf.int32, [None], name="state")
    R = tf.placeholder(tf.float32, [None], name="R")
    A = tf.placeholder(tf.int32, [None], name="action")
    exp_x = tf.one_hot(X, env.observation_space.n)
    with tf.variable_scope("model"):
        m = DiscreteModel(exp_x, env.action_space)
    m.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")

    log_policy = tf.nn.log_softmax(m.policy)
    R_e = tf.expand_dims(R, axis=1)
    value_loss = tf.reduce_sum(tf.square(m.value - R_e))
    advantage = tf.stop_gradient(R_e - m.value)
    expanded_a = tf.one_hot(A, env.action_space.n)
    policy_loss = -tf.reduce_sum(expanded_a * log_policy * advantage)
    entropy = - tf.reduce_sum(tf.nn.softmax(m.policy) * log_policy)
    tf.summary.scalar("model/entropy", entropy)
    tf.summary.scalar("model/policy_loss", policy_loss)
    tf.summary.scalar("model/value_loss", value_loss)

    loss = 0.25 * value_loss + policy_loss - 0.01 * entropy
    v_grads = tf.gradients(value_loss, m.vars)[2:]

    summary_op = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.GradientDescentOptimizer(1e-1).minimize(loss,
            global_step=global_step)

    sw = tf.summary.FileWriter(LOGDIR)
    sv = tf.train.Supervisor(logdir=LOGDIR,
            summary_writer=None,
            summary_op=None,
            save_model_secs=30)

    plt.ion()
    with sv.managed_session() as sess:
        sw.add_graph(sess.graph)
        iters = 0
        while True:
            done = False
            rew = []
            act = []

            o = env.reset()
            while not done:
                # env.render()
                a = sess.run(m.sample, feed_dict={X: [o]})
                new_o, r, done, i = env.step(a)

                estimated_R = r + 0.9 * sess.run(m.value, feed_dict={X: [o]})[0][0]
                s, _, g = sess.run([summary_op, train_op, v_grads],
                        feed_dict={X: [o], A: [a], R: [estimated_R]})
                sw.add_summary(s, global_step=sess.run(global_step))
                o = new_o

                iters += 1
                # print("Value estimation")
                if iters % 500 == 0:
                    states = np.arange(env.observation_space.n)
                    ps, vs, scheck = sess.run([m.policy, m.value, m.input],
                            feed_dict={X: states})
                    plt.subplot(121)
                    plt.title("value")
                    plt.imshow(np.reshape(vs, (4, 4)), cmap='hot',
                            interpolation='none')
                    plt.subplot(122)
                    plt.title("policy")
                    di = np.zeros((12, 12))
                    di[0,0] = 0
                    di[11,11] = 1
                    ps_softmax = np.exp(ps) / np.sum(np.exp(ps), axis=1)[:,None]
                    ps_softmax = np.reshape(ps_softmax, (4, 4, 4))
                    for ir, r in enumerate(range(1, 12, 3)):
                        for ic, c in enumerate(range(1, 12, 3)):
                            di[r, c - 1] = ps_softmax[ir, ic, 0]
                            di[r + 1, c] = ps_softmax[ir, ic, 1]
                            di[r, c + 1] = ps_softmax[ir, ic, 2]
                            di[r - 1, c] = ps_softmax[ir, ic, 3]
                    plt.imshow(di, cmap='hot', interpolation='none')
                    print("Updated graphic")
                    plt.pause(0.01)
            s = tf.Summary(value=[
                tf.Summary.Value(tag="reward", simple_value=sum(rew)),
                tf.Summary.Value(tag="length", simple_value=len(rew))
                ])
            sw.add_summary(s, global_step=sess.run(global_step))


def play_text_env(ENV='FrozenLake-v0'):
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
    play_taxi_env()

#!/usr/bin/env python3
from gridworld import GridWorld
from models import ICMModule, DiscreteModel
import sys
sys.path.append("..")

import gym
from utils import *
import time
import numpy as np
import matplotlib.pyplot as plt


def a3c_trainer(env, train, policy, summary_op = None, batch_size=32,
        should_stop=lambda: False):
    while not should_stop():
        obs = []
        rew = []
        act = []
        total_r = []
        o = env.reset()
        d = False
        while not d:
            obs.append(o)
            a = policy(o)
            act.append(a)
            new_o, done, r, i = env.step(a)
            total_r.append(r)
            rew.append(r)
            if summary_op is not None:
                summary_op(i)
            if len(obs) >= batch_size and not d:
                train(obs, rew, act, new_o, False)
                obs = []
                rew = []
                act = []
            o = new_o
        train(obs, rew, act, new_o, True)
        if summary_op is not None:
            summary_op({'game_length': len(total_r),
                'game_reward': sum(total_r)})


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
    tf.summary.scalar("icm/forward_loss", forward_loss / (bs - 1))
    # tf.summary.scalar("icm/inverse_loss", inverse_loss / (bs - 1))

    beta = 0.9
    loss = 0.25 * value_loss + policy_loss - 0.01 * entropy
    icm_loss = beta * forward_loss + (1 - beta) * inverse_loss

    summary_op = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.GradientDescentOptimizer(3e-4).minimize(loss,
            global_step=global_step)
    icm_train_op = tf.train.GradientDescentOptimizer(3e-3).minimize(icm_loss,
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
        while not sv.should_stop():
            iters = 0
            def train(obs, rew, act, last_o, terminal=False, gamma=0.9, curiosity=True,
                    mpl=True):
                if len(obs) == 0:
                    print("what")
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
                    states = np.arange(16)
                    v, po_r = sess.run([m.value, m.policy],
                            feed_dict={X: states})
                    po_r = np.exp(po_r) / np.exp(po_r).sum(axis=1)[:, None]
                    po_r = np.reshape(po_r, (4, 4, 4))
                    disp = np.zeros((6, 6))
                    disp[1:5, 1:5] = v.reshape((4, 4))
                    disp[5, 5] = 1
                    disp[5, 4] = -1
                    plt.subplot(121)
                    plt.title("value")
                    plt.imshow(disp, cmap='Greys', interpolation='none')

                    plt.subplot(122)
                    plt.title("policy")
                    pc = np.zeros((16, 16))
                    pc[0, 0] = 0
                    pc[11, 11] = 1
                    for ir, r in zip(range(4), range(1, 12, 3)):
                        for ic, c in zip(range(4), range(1, 12, 3)):
                            pc[r - 0, c - 1] = po_r[ir, ic, 0]
                            pc[r + 1, c + 0] = po_r[ir, ic, 1]
                            pc[r + 0, c + 1] = po_r[ir, ic, 2]
                            pc[r - 1, c - 0] = po_r[ir, ic, 3]
                    plt.imshow(pc, cmap='Greys', interpolation='none')
                    plt.suptitle("Global step: {}".format(
                        sess.run(global_step)))
                    plt.pause(0.01)
                iters += 1

            def policy(state):
                return sess.run(m.sample, feed_dict={X: [state]})

            def record_summary(info):
                v = [tf.Summary.Value(tag=k, simple_value=k)
                    for k, v in info.items()]
                sw.add_summary(tf.Summary(value=v),
                    global_step=sess.run(global_step))
            a3c_trainer(env, train, policy, record_summary, batch_size=5)



def play_text_env(ENV='gridworld'):
    if ENV == 'gridworld':
        env = GridWorld()
    else:
        env = gym.make(ENV)
    move_to_num = {'l': 0, 'd': 1, 'r': 2, 'u': 3}
    o = env.reset()
    while True:
        env.render()
        print("State: ", o)
        m = input().strip()
        o, r, d, _ = env.step(move_to_num[m])
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

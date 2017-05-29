import numpy as np
import tensorflow as tf
from utils import *
import gym
import time
import matplotlib.pyplot as plt


class CartpoleModel():
    def __init__(self, inp, output_space):
        if inp is not None:
            self.input = inp
        if isinstance(output_space, gym.spaces.Discrete):
            self.output_space = output_space
        else:
            raise NotImplementedError("Discrete only!")

        self._build_net(self.input)

    def _build_net(self, inp):
        net = inp
        net = tf.contrib.layers.flatten(net)
        self.value = tf.layers.dense(net, 1)
        net = tf.layers.dense(net, self.output_space.n,
                kernel_initializer=tf.uniform_unit_scaling_initializer(0.001))

        self.logits = net
        self.prob = tf.nn.softmax(self.logits)
        self.sample = tf.squeeze(tf.multinomial(self.logits, 1))


    def sample_action(self, sess, observation):
        act = sess.run(self.prob, feed_dict={self.input: observation})
        act = np.random.choice(act.size, p=act.squeeze())
        return act


class Trainer():
    def __init__(self, train_op, V, L):
        self.train_op = train_op
        self.V = V
        self.L = L

def guess_cartpole():
    env = gym.make('MountainCar-v0')
    w = np.zeros(env.observation_space.shape).astype('float32')
    print(w.shape)
    b = np.array(0).astype('float32')
    param = [w, b]
    def p(o):
        return np.sign(np.dot(o, w) + b) > 0

    def get_run(pw, pb, render=False):
        w, b = param
        w += pw
        b += pb
        o = env.reset()
        d = False
        R = 0
        while not d:
            if render:
                env.render()
                time.sleep(0.01)
            o, r, d, _ = env.step(p(o))
            R += r
        return R

    best_score = 0
    for i in range(1000):
        # get_run(0, 0, True)
        perts = [(np.random.normal(size=env.observation_space.shape).astype('float32'),
            np.random.normal()) for p in range(1000)]
        res = list(map(
            lambda p: (get_run(*p), p), perts))
        score, best = max(res, key=lambda v: v[0])
        if best_score < score:
            w += best[0]
            b += best[1]
            best_score = score
        print("Best score:", best_score)


def train_with_data(sess, model, trainer, data, last_reward=0, lmda=0.99,
        batch_size=512, gradients_debug=None):
    s, a, r, _ = data
    R = last_reward
    Rewards = []
    for rew in r[::-1]:
        R = lmda * R + rew
        Rewards = [R] + Rewards
    for i in range(0, len(s), batch_size):
        feed_dict = {model.input: s[i:i+batch_size],
            trainer.V: Rewards[i:i+batch_size],
            trainer.L: a[i:i+batch_size]}
        sess.run(trainer.train_op, feed_dict=feed_dict)
        if gradients_debug is not None:
            print(sess.run(gradients_debug, feed_dict=feed_dict))


def simple_a2c():
    env = gym.make("CartPole-v0")
    X = get_placeholder(env.observation_space)
    L = tf.placeholder(tf.int32, [None])
    V = tf.placeholder(tf.float32, [None])
    with tf.variable_scope("model"):
        m = CartpoleModel(X, env.action_space)
    v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")

    actions = tf.one_hot(L, env.action_space.n)

    log_logits = tf.nn.log_softmax(m.logits)
    value = tf.expand_dims(V, axis=1)

    pi_error = -tf.reduce_sum(
        tf.stop_gradient(V - m.value) * tf.reduce_sum(actions * log_logits,
            axis=1))
    value_error = 0.5 * tf.reduce_sum(tf.squared_difference(V, m.value))
    entropy = - tf.reduce_sum(m.prob * log_logits)
    error = pi_error + value_error - 0.001 * entropy
    g = tf.gradients(error, v)

    train_op = tf.train.AdamOptimizer(1e-2).minimize(error)
    init_op = tf.global_variables_initializer()

    trainer = Trainer(train_op, V, L)
    beat = False

    with tf.Session() as sess:
        sess.run(init_op)
        while True:
            data = run_env(env, sess, m.input, m.sample, render=True)
            R = sum(data[2])
            if R == 200:
                beat = True
            print("Total reward: %d" % R)
            train_with_data(sess, m, trainer, data, gradients_debug=None)
            # print(sess.run(m.prob, feed_dict={m.input: data[0]}))


if __name__ == '__main__':
    simple_a2c()

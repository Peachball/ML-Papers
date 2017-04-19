from __future__ import print_function
from utils import *
import gym
import argparse
import tensorflow as tf
import abc
import threading
import time
import numpy as np
import os
import shutil

class Model():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input = tf.placeholder(tf.float32, [None] + input_dim)
        self.summaries = []

    @abc.abstractmethod
    def train_op(self):
        pass

    @abc.abstractmethod
    def sample_action(self, obs):
        pass

class Trainer():
    @abc.abstractmethod
    def start_training(self):
        pass

    @abc.abstractmethod
    def stop_training(self):
        pass


def get_args():
    parser = argparse.ArgumentParser(description="Use DQN to play atari")

    parser.add_argument('--env', default='Pong-v0',
            help="Name of the environment to create")

    parser.add_argument('--threads', default=4,
            help="Number of worker threads to create")

    parser.add_argument('--device', default='/cpu:0')

    parser.add_argument('--logdir', default='tflogs/a2c')

    parser.add_argument("--save-interval", default=25)

    args = parser.parse_args()
    return args


class A2C(Model):
    def __init__(self, input_dim, out_dim, batch_size=32, isdiscrete=False):
        input_dim = list(input_dim)
        out_dim = list(out_dim)
        super(A2C, self).__init__(input_dim, out_dim[0])
        self.init_train_op = False
        self.data_queue = tf.RandomShuffleQueue(10000, 100, [tf.float32,
            tf.float32, tf.float32], shapes=[input_dim, out_dim, 1])
        self.train_data = self.data_queue.dequeue_many(batch_size)
        self.actions = tf.placeholder(tf.float32, [None] + out_dim,
                name="action")
        self.rewards = tf.placeholder(tf.float32, [None, 1], name="reward")

        self.isdiscrete = isdiscrete
        self.built_net = False
        self.train_op
        self.built_sample_action = False
        self.sample_action
        self._init_value = False
        self.value

    @property
    def train_op(self):
        if not self.init_train_op:
            im, a, R = self.train_data
            v = None
            if self.isdiscrete:
                v, a_l = self._build_net(im)
                a_p = tf.nn.softmax(a_l)
                self.policy_error = tf.reduce_mean(
                        (R - tf.stop_gradient(v)) * tf.log(
                            tf.reduce_sum(
                                tf.clip_by_value(a_p, 0.001,
                                    0.999) * a, axis=1)))
                self.summaries += [
                        tf.summary.scalar("Model entropy",
                            tf.reduce_mean(
                                tf.reduce_sum(-a_p * tf.log(a_p), axis=1)))]
            else:
                v, [a_m, a_v] = self._build_net(im)
                prob = tf.contrib.distributions.Normal(
                        mu = a_m, sigma = tf.nn.softplus(a_v))
                self.policy_error = tf.reduce_mean(
                        (R - tf.stop_gradient(v)) * tf.log(prob.pdf(a)))
            self.value_error = tf.reduce_mean(tf.squared_difference(R, v))
            self.summaries += [
                    tf.summary.scalar("Policy Error", self.policy_error),
                    tf.summary.scalar("Value Error", self.value_error)]

            v_opt = tf.train.RMSPropOptimizer(1e-4, epsilon=0.1)
            p_opt = tf.train.RMSPropOptimizer(1e-4, epsilon=0.1)

            self.v_train_op = v_opt.apply_gradients(
                    clip_gvs(
                        v_opt.compute_gradients(self.value_error), 10),
                    global_step=tf.train.get_global_step())
            self.p_train_op = p_opt.apply_gradients(
                    clip_gvs(p_opt.compute_gradients(self.policy_error), 10),
                    global_step=tf.train.get_global_step())

            self.init_train_op = True
        return [self.v_train_op, self.p_train_op]

    @property
    def sample_action(self):
        if self.built_sample_action:
            return self.sa_v
        self.built_sample_action = True
        if not self.isdiscrete:
            _, [a_m, a_v] = self._build_net(self.input)
            self.sa_v = tf.random_normal([self.output_dim]) * tf.nn.softplus(a_v) + a_m
            return self.sa_v
        else:
            _, a_l = self._build_net(self.input)
            self.sa_v = tf.clip_by_value(tf.nn.softmax(a_l), 1e-3, 1 - 1e-3)
            return self.sa_v

    @property
    def enqueue_op(self):
        return self.data_queue.enqueue_many(
                [self.input, self.actions, self.rewards])

    @property
    def value(self):
        if self._init_value:
            return self.q_value
        self.q_value, _ = self._build_net(self.input)
        return self.q_value

    @classmethod
    def from_env(cls, env):
        image_dim = env.observation_space.high.shape
        out_dim = []
        discrete = False
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            out_dim = [env.action_space.n]
            discrete = True
        else:
            out_dim = env.action_space.high.shape
            discrete = False
        return cls(image_dim, out_dim, isdiscrete=discrete)

    def _build_net(self, inp):
        net = None
        w_init = tf.uniform_unit_scaling_initializer(0.01)
        with tf.variable_scope("state_input", reuse=self.built_net):
            if len(self.input_dim) == 3:
                net = add_conv_layer(inp, [8, 8], 16, "conv_1", strides=[4, 4],
                        w_init=w_init)
                net = tf.nn.elu(net)
                net = add_conv_layer(net, [4, 4], 32, "conv_2", strides=[2, 2],
                        w_init=w_init)
                net = tf.nn.elu(net)
                net = flatten(net)
                net = add_layer(net, 256, "fc_3", w_init=w_init)
                net = tf.nn.elu(net)
            if len(self.input_dim) == 1:
                net = add_layer(inp, 128, "fc1")
                net = tf.nn.tanh(net)
                net = add_layer(net, 128, "fc2")
                net = tf.nn.tanh(net)
                net = add_layer(net, 128, "fc3")
                net = tf.nn.tanh(net)
                net = add_layer(net, 128, "fc4")
                net = tf.nn.tanh(net)

        if net is None:
            raise NotImplementedError("Net not assigned value")
        with tf.variable_scope("value", reuse=self.built_net):
            value = add_layer(net, 1, "value")

        with tf.variable_scope("action_probabilities", reuse=self.built_net):
            if self.isdiscrete:
                action_logits = add_layer(net, self.output_dim, "action_probs",
                        w_init=w_init)
                self.built_net = True
                return value, action_logits
            else:
                action_means = add_layer(net, self.output_dim, "policy_mean",
                        w_init=w_init)
                action_log_variance = add_layer(net, self.output_dim,
                        "policy_var",
                        w_init=w_init)
                self.built_net = True
                return value, [action_means, action_log_variance]

class FeudalNet():
    def __init__(self, input_space, output_space, horizon=64, goal_size=16,
            intrinsic_reward_influence=0.5, discount_factor=0.99,
            entropy_regularization=0.01):
        self.output_space = output_space
        self.input_space = input_space
        output_dim = None
        self.rewards = tf.placeholder(tf.float32, [None, None], name="rewards")
        if type(self.output_space) is gym.spaces.MultiDiscrete:
            output_dim = (self.output_space.high - self.output_space.low).sum()
            self.action = tf.placeholder(
                    tf.int32,
                    [None, None, self.output_space.low.shape[0]],
                    name="action_input")
        if type(self.output_space) is gym.spaces.Discrete:
            output_dim = self.output_space.n
            self.action = tf.placeholder(tf.int32, [None, None],
            name="action_input")
        self.output_dim = output_dim

        if output_dim is None:
            raise NotImplementedError("Cannot handle nondiscrete output spaces")
        if type(input_space) is not gym.spaces.Box:
            raise NotImplementedError(
                    "Cannot handle non box observation spaces yet")
        self.input_dim = input_space.shape
        self.input = tf.placeholder(tf.float32, [None, None] + list(input_space.shape),
                name="observation")
        self.initialized = []
        self._train_ops = {}
        self._action_ops = {}
        self._value_ops = {}
        self._states = {}
        self.horizon = horizon
        self.goal_size = goal_size
        self.ir_influence = intrinsic_reward_influence
        self.summaries = {}

        self.game_length = tf.placeholder(tf.float32, shape=())
        self.total_reward = tf.placeholder(tf.float32, shape=())
        # self.game_summaries = tf.summary.merge([
                # tf.summary.scalar("Game Length", self.game_length),
                # tf.summary.scalar("Total reward", self.total_reward)])
        self.discount = discount_factor
        self.entropy_reg = entropy_regularization

        self.train_op()

    def train_op(self, device=None):
        if device is None:
            device = '/cpu:0'
        if device in self.initialized:
            return self._train_ops[device]
        self.initialized += [device]

        with tf.device(device):
            if device not in self.summaries:
                self.summaries[device] = []
            p, [g, s], [v, v_m], [w_state, m_state] = self._build_net(self.input)
            prob = None
            expanded_label = None
            policy_error = None
            time_g = tf.transpose(g, [1, 0, 2])
            time_s = tf.transpose(s, [1, 0, 2])
            # time_s = [time, batch, state dimension]
            ind = tf.range(tf.shape(time_g)[0])
            intrinsic_reward = tf.scan(lambda a, i_s: tf.cond(
                tf.greater_equal(i_s[0], self.horizon),
                lambda: tf.reduce_mean(tf.scan(
                    lambda a, ii: self._cos(
                        i_s[1] - time_s[i_s[0] - ii],
                        time_g[i_s[0] - ii]),
                    tf.range(self.horizon),
                    initializer=tf.zeros_like(time_g[0,:,0])),
                    axis=0) + self.discount * a,
                lambda: tf.zeros_like(time_g[0,:,0])
                ),
                (ind[::-1], time_s[::-1]),
                initializer=tf.stop_gradient(v_m[:,-1])
                )[::-1]

            # Convert intrinisc reward back into [batch, timestep]
            intrinsic_reward = tf.check_numerics(
                    tf.transpose(intrinsic_reward, [1, 0]), "Intrinsic reward")

            value_error = tf.reduce_sum(tf.squared_difference(
                self.rewards + tf.stop_gradient(
                    self.ir_influence * intrinsic_reward), v))
            if type(self.output_space) is gym.spaces.MultiDiscrete:
                h = self.output_space.high
                l = self.output_space.low
                ind = 0
                ranges = []
                labels = []
                for i in range(l.shape[0]):
                    ranges.append(p[:,:,ind:(h[i] - l[i])])
                    labels.append(tf.one_hot(
                        self.action[:,:,ind:(h[i] - l[i])] - l[i],
                        h[i] - l[i]))
                prob = tf.concat(
                        [tf.nn.softmax(reg) for reg in ranges],
                        axis=-1)
                log_prob = tf.concat([tf.log(tf.reduce_sum(r * l, axis=-1))
                    for r, l in zip(ranges, labels)], axis=-1)
                policy_error = tf.reduce_sum(
                        -(self.rewards + tf.stop_gradient(self.ir_influence * intrinsic_reward - v)) *
                        tf.reduce_sum(log_prob, axis=-1))
            if type(self.output_space) is gym.spaces.Discrete:
                prob = tf.nn.softmax(p)
                entropy = tf.reduce_sum(-prob * tf.log(prob + 1e-8))
                self.summaries[device] += [tf.summary.scalar("model/entropy", entropy)]
                expanded_label = tf.one_hot(self.action, self.output_space.n)
                policy_error = tf.reduce_sum(
                        -tf.stop_gradient(
                            self.rewards + self.ir_influence * intrinsic_reward - v) *
                        tf.log(tf.reduce_sum(expanded_label * prob, axis=-1) +
                            1e-8)) - entropy * self.entropy_reg

            if prob is None:
                raise NotImplementedError()

            #[i, g, s, adv]
            manager_error = tf.scan(lambda a, pi: tf.cond(
                    tf.greater(tf.shape(time_s)[0], self.horizon + pi[0]),
                    lambda: -tf.reduce_mean(pi[3] * self._cos(
                        time_s[pi[0] + self.horizon] - pi[2],
                        pi[1])),
                    lambda: tf.constant(0, dtype=tf.float32)),
                (ind, time_g, time_s, tf.transpose(self.rewards - v_m, [1, 0])),
                    initializer=np.array(0.0).astype('float32'), back_prop=True)
            manager_error = tf.reduce_sum(manager_error)
            manager_value_error = tf.reduce_sum(
                    tf.squared_difference(self.rewards, v_m))

            self.summaries[device] += [
                    tf.summary.scalar("worker/policy error", policy_error),
                    tf.summary.scalar("worker/value error", value_error),
                    tf.summary.scalar("manager/error", manager_error),
                    tf.summary.scalar("manager/value error",
                        manager_value_error)
                ]
            reuse = getattr(self, '_init_optimizer', False)
            self._init_optimizer = True
            opt = tf.train.AdamOptimizer(1e-4)
            # r_opt = tf.train.RMSPropOptimizer(1e-5, epsilon=0.1)
            train_op = self._get_train_op(policy_error +
                    0.25 * value_error +
                    manager_error +
                    0.25 * manager_value_error,
                    opt, reuse=reuse)
            # p_opt = self._get_train_op(policy_error, opt)
            # v_opt = self._get_train_op(value_error, opt)
            # m_opt = self._get_train_op(manager_error, opt)
            # vm_opt = self._get_train_op(manager_value_error, opt)
            if prob is not None:
                self._action_ops[device] = [prob, w_state, m_state]
            else:
                raise NotImplementedError("Did not build MultiDiscrete yet")
            self._value_ops[device] = [v, v_m]
            self._states[device] = [w_state, m_state]
            self._train_ops[device] = [train_op]
            self.summaries[device] = tf.summary.merge(self.summaries[device])
            return self._train_ops[device]

    def _get_train_op(self, target, optimizer, reuse=None):
        with tf.variable_scope("gradients", reuse=reuse):
            gvs = optimizer.compute_gradients(target)
            clipped_gvs = [(tf.clip_by_norm(g, 5), v)
                    for g, v in gvs if g is not None]
            train_op = optimizer.apply_gradients(clipped_gvs,
                    global_step=tf.train.get_global_step())
            return train_op

    def get_value_op(self, device='/cpu:0'):
        if device in self._value_ops:
            return self._value_ops[device]
        self.train_op()
        return self._value_ops[device]

    def get_end_states(self, device='/cpu:0'):
        if device in self._states:
            return self._states[device]
        self.train_op()
        return self._states[device]

    def _cos(self, t1, t2, epsilon=1e-8):
        assert len(t1.get_shape()) == len(t2.get_shape())
        prod = tf.matmul(t1, t2, transpose_b=True)
        mag1 = tf.sqrt(tf.reduce_sum(tf.square(t1), axis=-1) + epsilon)
        mag2 = tf.sqrt(tf.reduce_sum(tf.square(t2), axis=-1) + epsilon)
        normalized = tf.reduce_sum(prod, axis=-1) / (mag1 * mag2)
        return normalized

    def _multidiscrete_error(self, p, l):
        pass

    def _build_net(self, inp):
        """
            Input dimensions are assumed to be: [batch size, time steps, image dim]
        """
        reuse = "built net" in self.initialized
        if not reuse:
            self.initialized += ["built net"]
        with tf.variable_scope("feudalnet", reuse=reuse):
            i_s = tf.shape(inp, name="original_input_shape")
            reshaped_images = tf.reshape(
                    inp,
                    [i_s[0] * i_s[1]] + list(self.input_dim),
                    name="reshape_image_for_perception")
            z_t = self.perception_function(reshaped_images)
            z_t = tf.reshape(z_t, [tf.shape(inp)[0], tf.shape(inp)[1],
                int(z_t.get_shape()[-1])],
                name="return_reshaped_image_to_batch_time")
            value = self._value(z_t)
            manager_value = self._m_value(z_t)
            s_t = self.manager_space(z_t)
            worker_output, w_states = self.worker_rnn(z_t,
                    batch_size=tf.shape(z_t)[0])
            goals, m_states = self.manager_rnn(s_t, batch_size=tf.shape(s_t)[0])
            time_major_goals = tf.transpose(goals, [1, 0, 2])
            g = self.embed_goal(time_major_goals)

            # Convert goals back to [batch, time, dim]
            g = tf.transpose(g, [1, 0, 2])
            w_t = tf.stop_gradient(g)
            flattened_w_t = tf.reshape(w_t,
                    [
                        tf.shape(w_t)[0] * tf.shape(w_t)[1],
                        int(w_t.get_shape()[-1])
                        ], name="w_t_to_flat")
            flat_proj = add_layer(flattened_w_t, self.goal_size,
                    "goal_projection", biases=False)
            w_t = tf.reshape(flat_proj,
                    [
                        tf.shape(w_t)[0],
                        tf.shape(w_t)[1],
                        int(w_t.get_shape()[-1])
                        ], name="flat_to_w_t")
            p = self._policy(worker_output, w_t)
            return p, [goals, s_t], [value, manager_value], [w_states, m_states]


    def _m_value(self, inp):
        with tf.variable_scope("manager_value"):
            s = tf.shape(inp)
            ld = int(inp.get_shape()[-1])
            flattened = tf.reshape(inp, [s[0] * s[1], ld])
            net = add_layer(flattened, 256, "fc1")
            net = tf.nn.elu(net)
            net = add_layer(net, 1, "fc2")
            return tf.reshape(net, [s[0], s[1]])

    def _value(self, inp):
        with tf.variable_scope("worker_value"):
            s = tf.shape(inp)
            ld = int(inp.get_shape()[-1])
            flattened = tf.reshape(inp, [s[0] * s[1], ld])
            net = add_layer(flattened, 256, "fc1")
            net = tf.nn.elu(net)
            net = add_layer(net, 1, "fc2")
            return tf.reshape(net, [s[0], s[1]])

    def embed_goal(self, goals):
        """
            Expects goals to be in dimension: [time, batch, goals]
        """
        o_s = tf.shape(goals)
        if "initial_goals" not in self.initialized:
            self.initial_goal = tf.zeros_like(goals[0])
        else:
            self.initialized += ["initial_goals"]
        def accumulate(a, v):
            i, r = v
            return tf.cond(
                    tf.less(i, self.horizon + 1),
                    lambda: a + r,
                    lambda: a + r - goals[i - self.horizon - 1])
        ind = tf.range(tf.shape(goals)[0])
        v = tf.scan(accumulate, (ind, goals), back_prop=False,
            initializer=self.initial_goal)
        return v

    def perception_function(self, inp):
        if len(inp.get_shape()) == 4:
            net = tf.image.resize_images(inp, [128, 128])
            tf.summary.image(net)
            net = add_conv_layer(net, [8, 8], 16, "conv1", strides=[4, 4])
            net = tf.nn.elu(net)
            net = add_conv_layer(net, [8, 8], 32, "conv2", strides=[4, 4])
            net = tf.nn.elu(net)
            net = flatten(net)
            return net
        raise NotImplementedError("Only can handle images currently")

    def manager_space(self, inp):
        with tf.variable_scope("manager_transformation"):
            o_s = tf.shape(inp)
            last_dim = int(inp.get_shape()[-1])
            net = tf.reshape(inp, [o_s[0] * o_s[1], last_dim])
            net = tf.nn.elu(add_layer(net, self.goal_size, 'fc'))
            net = tf.reshape(net, [o_s[0], o_s[1], self.goal_size])
            return net

    def _policy(self, U_t, w_t):
        """
            expects dimenions to be:
                U_t = [batch, time, actions, goal size]
                w_t = [batch, time, goal size]
        """
        return tf.squeeze(tf.matmul(U_t, tf.expand_dims(w_t, axis=-1)), axis=-1)

    def manager_rnn(self, inp, batch_size=1):
        """
            Assumes input is of the form: [batch size, time steps, features]
        """
        reuse = False
        if hasattr(self, '_init_manager_rnn'):
            reuse = True
        self._init_manager_rnn = True
        with tf.variable_scope("manager_rnn"):
            lstm = tf.contrib.rnn.LSTMCell(self.goal_size, use_peepholes=True,
                    reuse=reuse)
            self.manager_init_state = lstm.zero_state(batch_size, tf.float32)
            outputs, state = tf.nn.dynamic_rnn(lstm, inp, dtype=tf.float32,
                    initial_state=self.manager_init_state)
            normalized = outputs / tf.expand_dims(tf.sqrt(
                    tf.reduce_sum(tf.square(outputs), axis=-1)), axis=-1)
            return normalized, state

    def worker_rnn(self, inp, batch_size=1):
        """
            Assumes input is of the form: [batch size, time steps, features]
        """
        reuse = False
        if hasattr(self, '_init_worker_rnn'):
            reuse = True
        self._init_worker_rnn = True
        output_size = None
        if type(self.output_space) is gym.spaces.MultiDiscrete:
            output_size = (self.output_space.high - self.output_space.low).sum()
        if type(self.output_space) is gym.spaces.Discrete:
            output_size = self.output_space.n

        if output_size is None:
            raise NotImplementedError(
                    "Unable to handle non discrete output spaces")
        with tf.variable_scope("worker_rnn"):
            lstm = tf.contrib.rnn.LSTMCell(output_size * self.goal_size,
                    use_peepholes=True, reuse=reuse)
            self.worker_init_state = lstm.zero_state(batch_size, tf.float32)
            output, state = tf.nn.dynamic_rnn(lstm, inp, dtype=tf.float32,
                    initial_state=self.worker_init_state)
            s = tf.shape(output)
            return tf.reshape(output, [s[0], s[1], output_size, self.goal_size]), state

    def get_action(self, device='/cpu:0'):
        if device in self._action_ops:
            return self._action_ops[device]

        self.train_op(device=device)
        return self._action_ops[device]

    @classmethod
    def from_env(cls, env):
        return cls(env.observation_space, env.action_space)


class DataLoader(Trainer):
    def __init__(self, model, env, sess, discount_factor=0.99, sw=None):
        self.model = model
        self.env = env
        self.discount_factor = discount_factor
        self.sess = sess
        self.debugging_info = ["Game length", "Total Reward"]
        self.gl = tf.placeholder(tf.float32)
        self.tr = tf.placeholder(tf.float32)
        self.summaries = tf.summary.merge([
                tf.summary.scalar("Game length", self.gl),
                tf.summary.scalar("Total reward", self.tr)])
        self.sw = sw

    def _train_loop(self, max_iters=10000, render=False):
        sess = self.sess
        env = self.env
        while self.train:
            r = []
            o = []
            a = []
            d = False
            c_o = env.reset()
            rew = 0
            iters = 0
            while not d and iters < max_iters and self.train:
                iters += 1
                if render:
                    env.render()
                o += [c_o]
                r += [rew]
                action = sess.run(self.model.sample_action,
                        feed_dict={self.model.input: c_o[None,:]})
                if type(env.action_space) != gym.spaces.discrete.Discrete:
                    c_o, rew, d, _ = env.step(action.squeeze())
                    a += [action.squeeze()]
                else:
                    c = np.random.choice(
                            env.action_space.n,
                            p = action.squeeze())
                    c_o, rew, d, _ = env.step(c)
                    a_b = np.zeros_like(action.squeeze())
                    a_b[c] = 1
                    a += [a_b]

            if self.sw is not None:
                s = sess.run(self.summaries, feed_dict={self.tr: sum(r),
                    self.gl: iters})
                self.sw.add_summary(s)

            if d:
                R = 0
            else:
                R = sess.run(m.value, feed_dict={m.input: np.array(o)[-2:-1]})
            R_all = []
            for c_o, action, rew in zip(o, a, r):
                R = rew + self.discount_factor * R
                R_all = [R] + R_all

            sess.run(self.model.enqueue_op,
                    feed_dict = {
                        self.model.input: np.array(o),
                        self.model.action: np.array(a),
                        self.model.rewards: np.array(R_all)[:,None]})

    def start_training(self, render=False):
        self.train_thread = threading.Thread(target=self._train_loop,
                kwargs={'render': render})
        self.train = True
        self.train_thread.start()

    def stop_training(self):
        self.train = False
        self.train_thread.join()


def run_A2C():
    args = get_args()
    env = gym.make(args.env)
    envs = [gym.make(args.env) for i in range(args.threads)]

    global_step = tf.Variable(0, trainable=False, name='global_step')
    m = A2C.from_env(env)

    sw = tf.summary.FileWriter(args.logdir)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(args.logdir)
        try:
            saver.restore(sess, ckpt)
        except:
            print("\n\n\n\n\nWARNING: ON DELETE PAST MODE\n\n\n\n\n")
            shutil.rmtree(args.logdir)
            os.mkdir(args.logdir)
            sw.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())

        dls = [DataLoader(m, e, sess, sw=sw) for e in envs]
        for d in dls[:-1]:
            d.start_training()
        dls[-1].start_training(render=True)

        save_interval = time.clock()
        while True:
            try:
                _, ve, pe, s = sess.run([m.train_op, m.value_error, m.policy_error,
                    m.summaries])
                sw.add_summary(s)
                print("Value Error: {}\nPolicy Error: {}".format(ve, pe))
                if time.clock() - save_interval > args.save_interval:
                    saver.save(sess,
                            os.path.join(args.logdir, 'model'),
                            global_step=global_step.eval())
            except KeyboardInterrupt:
                exit()


def sample_multidiscrete(p, action_space):
    assert type(action_space) is gym.spaces.MultiDiscrete
    h = action_space.high
    l = action_space.low
    actions = []
    ind = 0
    for i in range(l.shape[0]):
        actions.append(np.random.choice(h[i] - l[i], p=p[ind:(h[i] - l[i])]))
        ind += h[i] - l[i]

    return actions


def sample_discrete(p, action_space):
    assert type(action_space) is gym.spaces.Discrete
    return np.random.choice(action_space.n, p=p)


def sample(p, action_space):
    if type(action_space) is gym.spaces.Discrete:
        return sample_discrete(p.squeeze(), action_space)
    if type(action_space) is gym.spaces.MultiDiscrete:
        return sample_multidiscrete(p.squeeze(), action_space)
    raise NotImplementedError()

def trainer(model, sess, sw, env, coord, device, gs):
    MAX_SEQUENCE_LENGTH = 400
    while not coord.should_stop():
        o = env.reset()
        states = []
        rewards = []
        actions = []
        int_states = []
        done = False
        w_s = None
        m_s = None
        r = 0
        while not done and not coord.should_stop():
            int_states += [(w_s, m_s)]
            if w_s is None:
                [p, w_s, m_s] = sess.run(model.get_action(device=device),
                        feed_dict={model.input: o[None,None]})
            else:
                [p, w_s, m_s] = sess.run(model.get_action(device=device),
                        feed_dict={model.input: o[None,None],
                            model.worker_init_state: w_s,
                            model.manager_init_state: m_s})
            a = sample(p, env.action_space)
            states += [o]
            actions += [a]
            o, r, done, info = env.step(a)
            rewards += [r]

        total_reward = []
        if done:
            R = r
        else:
            R = sess.run(model.get_value_op(device=device),
                    feed_dict={model.input: o[None,None],
                               model.worker_init_state: w_s,
                               model.manager_init_state: m_s})[0].squeeze()

        # Store summaries
        game_summaries = tf.Summary()
        game_summaries.value.add(tag="game_length", simple_value=len(rewards))
        game_summaries.value.add(tag="total_reward", simple_value=sum(rewards))
        sw.add_summary(game_summaries, global_step=sess.run(gs))

        for r_t in rewards[::-1]:
            total_reward += [model.discount * R + r_t]
        total_reward = np.array(total_reward[::-1])
        print("Total Reward: {}\nGame Length: {}"
                .format(sum(rewards), len(rewards)))


        # Training with sequence lengths of 400
        s_m = np.array(states)
        a_m = np.array(actions)
        for i in range(0, len(s_m), MAX_SEQUENCE_LENGTH):
            end = i + MAX_SEQUENCE_LENGTH
            feed_dict = {
                    model.input: s_m[None,i:end],
                    model.rewards: total_reward[None,i:end],
                    model.action: a_m[None,i:end],
                    }
            w_s, m_s = int_states[i]
            if w_s is not None:
                feed_dict.update({
                    model.worker_init_state: w_s,
                    model.manager_init_state: m_s})
            _, _, s  = sess.run([
                model.train_op(device=device),
                model.get_action(device=device),
                model.summaries[device]],
                feed_dict=feed_dict)
            sw.add_summary(s, global_step=sess.run(gs))

def run_feudal():
    OUTPUT_LOG_DIR="tflogs/fun"
    ENV="Pong-v0"
    CPU_CORES = 8
    # import ppaquette_gym_doom
    env = gym.make(ENV)
    global_step = tf.Variable(0, name="global_step")
    m = FeudalNet.from_env(env)
    [m.train_op("/cpu:{}".format(i)) for i in range(CPU_CORES)] # Initialize train ops
    sw = tf.summary.FileWriter(OUTPUT_LOG_DIR)
    coord = tf.train.Coordinator()
    saver = tf.train.Saver()
    config = tf.ConfigProto(
            device_count = {"CPU": 8, "GPU": 1},
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        ckpt = tf.train.latest_checkpoint(OUTPUT_LOG_DIR)
        try:
            saver.restore(sess, ckpt)
        except:
            sess.run(tf.global_variables_initializer())
            print("Did not init previous checkpoint")
            sw.add_graph(sess.graph)

        envs = [gym.make(ENV) for i in range(CPU_CORES)]
        train_threads = [threading.Thread(
            target=trainer,
            args=(m, sess, sw, e, coord, '/cpu:{}'.format(i), global_step))
                for (i, e) in enumerate(envs)]

        for t in train_threads:
            t.start()

        while True:
            try:
                time.sleep(30)
                saver.save(sess, os.path.join(OUTPUT_LOG_DIR, "model"),
                        global_step=tf.train.get_global_step().eval())
            except KeyboardInterrupt:
                coord.request_stop()
                coord.join(train_threads)
                exit()


if __name__ == '__main__':
    run_feudal()

import tensorflow as tf
from utils import *
import numpy as np
import argparse
import sys
import gym
import queue
import threading
import os
try:
    import gym_ple
except ImportError:
    pass

class ICM():
    def __init__(self, inp, action, action_space):
        """
        params
        state: Must be a tensor of [batch_size, state_dim] that are all from the
            same run
        action: must be one_hot tensor of [batch_size, state_dim]
        """
        self.input = inp
        self.action = action
        self.action_space = action_space
        self.build_net()

    def build_net(self):
        with tf.variable_scope("state_rep"):
            self.state = self.state_rep(self.input)
        with tf.variable_scope("forward"):
            net = tf.concat([self.state[:-1], self.action[:-1]], axis=1)
            net = tf.layers.dense(net, 256, name="fc1")
            net = tf.nn.elu(net)
            net = tf.layers.dense(net, 256, name="fc2")
            net = tf.layers.dense(net, int(list(self.state.get_shape())[-1]),
                    name="out",
                    kernel_initializer=tf.uniform_unit_scaling_initializer(0.01))
            net = tf.nn.elu(net)
            self.state_prediction = net
            self.r_i = 0.5 * tf.norm(self.state[1:] - net, axis=1)
        with tf.variable_scope("inverse"):
            net = tf.concat([self.state[:-1], self.state[1:]], axis=1)
            net = tf.layers.dense(net, 256, name="fc1")
            net = tf.nn.elu(net)
            net = tf.layers.dense(net, 256, name="fc2")
            net = tf.nn.elu(net)
            if type(self.action_space) is gym.spaces.Discrete:
                net = tf.layers.dense(net, self.action_space.n, name="out")
                self.pred_logits = net
                self.pred_prob = tf.nn.softmax(net)
            else:
                raise NotImplementedError(
                        "Did not implemented space {}".format(
                            type(self.action_space)))

    def state_rep(self, inp):
        if len(list(inp.get_shape())) == 4:
            net = inp
            net = tf.image.resize_images(net, [96, 96])
            net = tf.layers.conv2d(net, 32, [8, 8], strides=(4, 4), name="conv1")
            net = tf.nn.elu(net)
            net = tf.layers.conv2d(net, 64, [4, 4], strides=(2, 2), name="conv2")
            net = tf.nn.elu(net)
            net = tf.contrib.layers.flatten(net)
        elif len(list(inp.get_shape())) == 2:
            net = inp
            net = tf.layers.dense(net, 256, name="fc1")
            net = tf.nn.elu(net)
            net = tf.layers.dense(net, 256, name="fc2")
            net = tf.nn.elu(net)
        else:
            raise NotImplementedError("Unable to process your weird tensor...")
        net = tf.layers.dense(net, 256, name="out",
                kernel_initializer=tf.uniform_unit_scaling_initializer(0.01))
        net = tf.nn.elu(net)
        return net


class PolicyValueModel():
    def __init__(self, inp, output_space, use_lstm=False):
        self.input = inp
        self.output_space = output_space

        act, val = self.build_net(self.input, use_lstm=use_lstm)
        self.action = act
        self.value = val

    def build_net(self, inp, use_lstm=False):
        k_init = tf.contrib.layers.xavier_initializer()
        net = inp
        if len(inp.get_shape()) == 4:
            # Images
            net = tf.image.resize_images(net, [96, 96])
            net = tf.layers.conv2d(net, 32, [8, 8], strides=(4, 4),
                    name='conv1',
                    kernel_initializer=k_init)
            net = tf.nn.elu(net, name='nl1')
            net = tf.layers.conv2d(net, 64, [4, 4], strides=(2, 2),
                    name='conv2', kernel_initializer=k_init)
            net = tf.nn.elu(net, name='nl2')
            net = tf.contrib.layers.flatten(net)
        else:
            net = tf.layers.dense(net, 256, name='fc1',
                    kernel_initializer=k_init)
            net = tf.nn.elu(net)
            net = tf.layers.dense(net, 256, name='fc2',
                    kernel_initializer=k_init)
            net = tf.nn.elu(net)
        net = tf.layers.dense(net, 256, name='fc3', kernel_initializer=k_init)
        self.state_embedding = net
        net = tf.nn.elu(net)
        value = tf.layers.dense(net, 1, name='value_output',
                kernel_initializer=tf.uniform_unit_scaling_initializer(0.1)) # [batch_size, 1]

        action = None
        if type(self.output_space) == gym.spaces.Discrete:
            action = tf.layers.dense(net, self.output_space.n,
                    name='action_logits',
                    kernel_initializer=tf.uniform_unit_scaling_initializer(0.01))
            self.logits = action
            action = tf.nn.softmax(action)

        if action is None:
            raise NotImplementedError(
                    "Did not implement nondiscrete action spaces")
        return (action, value)


class LSTMModel():
    """
        Policy that uses ICM (internal curiosity module) to help train
    """
    def __init__(self, X, action_space):
        self.input = X
        self.action_space = action_space
        self.build_net(X)

    def build_net(self, inp):
        k_init = tf.contrib.layers.xavier_initializer()
        net = inp
        if len(list(inp.get_shape())) > 2:
            net = tf.image.resize_images(net, [96, 96])
            net = tf.layers.conv2d(net, 32, [8, 8], strides=(4, 4),
                    name="conv1", kernel_initializer=k_init)
            net = tf.nn.elu(net)
            net = tf.layers.conv2d(net, 64, [4, 4], strides=(2, 2),
                    name="conv2", kernel_initializer=k_init)
            net = tf.nn.elu(net)
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 256, name="reshape")
            net = tf.nn.elu(net)
        else:
            net = tf.layers.dense(net, 256, name="fc1")
            net = tf.layers.dense(net, 256, name="fc2")

        time_dim_net = tf.expand_dims(net, axis=0)
        lstm = tf.contrib.rnn.LSTMCell(256)
        self.initial_state = lstm.zero_state(1, tf.float32)
        time_dim_outputs, self.state = tf.nn.dynamic_rnn(lstm, time_dim_net,
                initial_state=self.initial_state)
        net = tf.squeeze(time_dim_outputs, axis=0)
        self.state_rep = net

        if type(self.action_space) is gym.spaces.Discrete:
            self.logits = tf.layers.dense(net, self.action_space.n,
                    name="action",
                    kernel_initializer=tf.uniform_unit_scaling_initializer(0.1))
            self.prob = tf.nn.softmax(self.logits)
            self.sample_action = tf.squeeze(tf.multinomial(self.logits, 1))
            self.value = tf.layers.dense(net, 1, name="value")

        else:
            raise NotImplementedError("Action space: {} not implemented",
                    type(self.action_space))


class ICMModel():
    def __init__(self, X, action, env):
        self.input = X
        self.m = m = LSTMModel(X, env.action_space)
        expanded_action = tf.one_hot(action, env.action_space.n,
                dtype=tf.float32)
        with tf.variable_scope("curiosity"):
            self.icm = ICM(X, expanded_action, env.action_space)
        self.sample_action = m.sample_action
        self.value = m.value


class OnlineTrainer():
    def __init__(self, model, env, sess, stateful=False):
        self.model = model
        self.env = env
        self.sess = sess
        self.stateful = stateful
        self.data = queue.Queue(10)

    def start_runner(self):
        self.stop = False
        t = threading.Thread(target=self.get_run, args=())
        t.start()

    def data_loader(self):
        while not self.stop:
            self.get_run()

    def get_run(self, max_len=10**10, render=False):
        states = []
        actions = []
        rewards = []
        m = self.model
        o = self.env.reset()
        d = False
        r = 0
        game_length = 0
        score = 0
        while not d and len(states) < max_len:
            if render:
                self.env.render()
            act_prob = self.sess.run(m.action, feed_dict={m.input: o[None]})
            act = None
            if type(self.env.action_space) == gym.spaces.Discrete:
                act = np.random.choice(self.env.action_space.n,
                        p=act_prob.squeeze())

            states += [o]
            actions += [act]
            o, r, d, _ = self.env.step(act)
            rewards += [r]
            score += r
            game_length += 1
            if len(states) >= 32:
                self.data.put((states, actions, rewards, d, {}))
                states = []
                actions = []
                rewards = []

        self.data.put((states, actions, rewards, d, {'score': score, 'length':
            game_length}))

    def get_data(self, length):
        states = []
        actions = []
        rewards = []
        d = False
        i = {}
        while len(states) + 32 <= length:
            s, a, r, d, i = self.data.get()
            states += s
            actions += a
            rewards += r
            if d:
                break
        i.update({'s': states,
                    'a': actions,
                    'r': rewards,
                    'terminal': d})

        return i

    def data_gen(self, get_action, batch_size=32):
        while True:
            d = False
            states = []
            actions = []
            rewards = []
            o = self.env.reset()
            gl = 0
            tr = 0
            while not d:
                states.append(o)
                a = get_action(o)
                o, r, d, i = self.env.step(a)
                tr += r
                gl += 1
                actions.append(a)
                rewards.append(r)
                if len(states) >= 32:
                    yield {'s': states,
                            'a': actions,
                            'r': rewards,
                            'terminal': d}
                    states = []
                    actions = []
                    rewards = []
            yield {'s': states,
                    'a': actions,
                    'r': rewards,
                    'terminal': d,
                    'game_length': gl,
                    'score': tr}


def stateful_generator(env, policy, batch_size=32):
    while True:
        o = env.reset()
        obs = []
        rew = []
        act = []
        d = False
        state = None
        total_reward = 0
        total_length = 0
        s = None
        initial_state = None
        ps = None
        while True:
            obs.append(o)
            ps = s
            a, s = policy(o, s)
            o, r, d, i = env.step(a)
            act.append(a)
            rew.append(r)
            total_reward += r
            total_length += 1

            if d:
                break

            if len(act) >= batch_size:
                yield {'s': np.array(obs), 'a': np.array(act), 'r':
                        rew, 't': d, 'is': initial_state, 'ps': ps}
                obs = []
                act = []
                rew = []

        yield {'s': np.array(obs), 'a': np.array(act), 'r': rew,
                't': d, 'is': initial_state, 'ps': ps, 'tr': total_reward,
                'len': total_length}



def _parse_distributed_a3c():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int)
    parser.add_argument('--cluster_file', type=str)
    parser.add_argument('--env', type=str, default='Pong-v0')
    parser.add_argument('--type', type=str)
    parser.add_argument('--logdir', type=str, default='./tflogs/icm')

    return parser.parse_args()


def _parse_cluster(filepath):
    f = open(filepath, 'r')
    cluster = {'ps': [], 'worker': []}
    for l in f:
        loc, ntype = l.strip().split(' ')
        assert(ntype in ['ps', 'worker'])
        cluster[ntype] += [loc]
    return tf.train.ClusterSpec(cluster)


def collect_train_op(R_p, A, local_model, env, global_step):
    local_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='local')
    global_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='global')

    tf.summary.histogram("correct_rewards", R_p)
    R = R_p
    A_oh = tf.one_hot(A, env.action_space.n)
    tf.summary.histogram("model/value", local_model.value)
    variance = tf.squared_difference(local_model.value,
            tf.reduce_mean(local_model.value))
    r_var = tf.squared_difference(R, tf.reduce_mean(R))
    tf.summary.scalar("model/value_stddev",
        tf.sqrt(tf.reduce_mean(variance)))
    tf.summary.scalar("model/label_stddev",
            tf.sqrt(tf.reduce_mean(r_var)))
    tf.summary.scalar("model/value_norm", tf.global_norm(local_var))

    tf.summary.histogram("model/action", local_model.action)

    # pretty much copied off of the openai universe starter agent
    policy_error = tf.reduce_sum(
            -(R - tf.stop_gradient(local_model.value)) *
            tf.expand_dims(tf.reduce_sum(tf.nn.log_softmax(local_model.logits) * A_oh,
                axis=-1), axis=1))
    value_error = tf.reduce_sum(
            tf.squared_difference(R, local_model.value))
    entropy = tf.reduce_sum(
            tf.reduce_sum(
                -local_model.action *
                tf.nn.log_softmax(local_model.logits), axis=-1))
    total_error = policy_error + 0.25 * value_error - 0.01 * entropy
    batch_size = tf.cast(tf.shape(A)[0], tf.float32)
    tf.summary.scalar("error/Total Error", total_error / batch_size)
    tf.summary.scalar("error/value", value_error / batch_size)
    tf.summary.scalar("error/policy", policy_error / batch_size)
    tf.summary.scalar("Entropy", entropy / tf.cast(batch_size, tf.float32))
    opt = tf.train.AdamOptimizer(3e-5)
    g = [grad
            for grad in tf.gradients(total_error, local_var)]
    g, global_norm = tf.clip_by_global_norm(g, 1e10)
    tf.summary.histogram("Gradient values",tf.concat(
            [tf.reshape(grad, [-1]) for grad in g], 0))
    grad_1d = tf.concat([tf.reshape(grad, [-1]) for grad in g], 0)
    var_1d = tf.concat([tf.reshape(vv, [-1]) for vv in local_var], 0)
    tf.summary.scalar("gradients/norm", global_norm)
    gvs = list(zip(g, global_var))
    inc_global_step = global_step.assign_add(tf.shape(A)[0])
    train_op = tf.group(opt.apply_gradients(gvs), inc_global_step)
    return train_op


def distributed_a3c():
    args = _parse_distributed_a3c()

    cluster = _parse_cluster(args.cluster_file)


    if  args.type == 'ps':
        c = tf.ConfigProto(device_filters=['/job:ps'])
        server = tf.train.Server(
                cluster,
                job_name=args.type,
                task_index=args.task,
                config=c)
        server.join()

    else:
        c = tf.ConfigProto(
                device_filters=['/job:ps',
                    '/job:worker/task:{}'.format(args.task)],
                inter_op_parallelism_threads=2,
                intra_op_parallelism_threads=1)
        server = tf.train.Server(
                cluster,
                job_name=args.type,
                task_index=args.task,
                config=c)
        env = gym.make(args.env)
        with tf.device(
                tf.train.replica_device_setter(ps_device='/job:ps',
                    worker_device='/job:worker/task:{}/cpu:0'.format(args.task),
                    cluster=cluster)):
            X = get_placeholder(env.observation_space)
            global_step = tf.Variable(0, trainable=False, name='global_step')
            with tf.variable_scope('global'):
                global_model = PolicyValueModel(X, env.action_space)

            with tf.device('/job:worker/task:{}/cpu:0'.format(args.task)):
                with tf.variable_scope('local'):
                    local_model = PolicyValueModel(X, env.action_space)
            local_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope='local')
            global_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope='global')
            sync = [v1.assign(v2) for (v1, v2) in zip(local_var, global_var)]

            R_p = tf.placeholder(tf.float32, [None, 1])
            A = tf.placeholder(tf.int32, [None])
            with tf.name_scope("training"):
                train_op = collect_train_op(R_p, A, local_model, env, global_step)
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver(global_var)
            global_init_op = tf.global_variables_initializer()
            init_op = tf.variables_initializer(local_var)
        sw = tf.summary.FileWriter('tflogs/a3c/t{}'.format(args.task))

        def init_fn(sess):
            sess.run(global_init_op)

        sv = tf.train.Supervisor(
                is_chief=(args.task==0),
                logdir='tflogs/a3c/master',
                global_step=global_step,
                summary_writer=sw,
                local_init_op=init_op,
                init_op=global_init_op,
                init_fn=init_fn,
                ready_op=tf.report_uninitialized_variables(global_var),
                summary_op=None)

        config = tf.ConfigProto(
                device_filters=['/job:worker/task:{}'.format(args.task),
                    '/job:ps'])
        with sv.managed_session(server.target, config=config) as sess:
            sw.add_graph(sess.graph)
            trainer = OnlineTrainer(local_model, env, sess, stateful=False)
            def get_action(obs):
                a = sess.run(local_model.action, feed_dict={local_model.input:
                    obs[None,]})
                return np.random.choice(a.size, p=a.squeeze())
            data_generator = trainer.data_gen(get_action)
            while not sv.should_stop():
                sess.run(sync)
                run = next(data_generator)
                if 'game_length' in run:
                    print("Played game of length {}".format(run['game_length']))
                if run['terminal']:
                    R = 0
                else:
                    R = sess.run(local_model.value,
                            feed_dict={X: run['s'][-1:]}).squeeze()
                Rewards = []
                lmd = 0.99
                for r in run['r'][::-1]:
                    R = lmd * R + r
                    Rewards = [R] + Rewards

                assert(len(run['s']) == len(run['a']) == len(Rewards))
                if 'game_length' in run:
                    sw.add_summary(
                            get_scalar_summary("game/length",
                                run['game_length']),
                            sess.run(global_step))
                if 'score' in run:
                    sw.add_summary(
                            get_scalar_summary("game/Score", run['score']),
                            sess.run(global_step))
                while len(Rewards) != 0:
                    sess.run(sync)
                    batch_size = 1024
                    _, summ, v = sess.run([
                            train_op,
                            summary_op, local_model.value],
                            feed_dict={X: np.array(run['s'])[:batch_size],
                                       R_p: np.array(Rewards)[:batch_size, None],
                                       A: np.array(run['a'])[:batch_size]})
                    run['s'] = run['s'][batch_size:]
                    Rewards = Rewards[batch_size:]
                    run['a'] = run['a'][batch_size:]
                    gs = sess.run(global_step)
                    if args.task == 0:
                        sw.add_summary(summ, gs)


def distributed_icm():
    args = _parse_distributed_a3c()
    cluster = _parse_cluster(args.cluster_file)
    env = gym.make(args.env)

    if args.type == 'ps':
        c = tf.ConfigProto(device_filters=['/job:ps'])
        server = tf.train.Server(cluster, job_name=args.type,
                task_index=args.task, config=c)
        server.join()
    elif args.type == 'worker':
        worker_device = '/job:worker/cpu:0/task:{}'.format(args.task)
        c = tf.ConfigProto(device_filters=['/job:ps', worker_device])
        server = tf.train.Server(cluster, job_name=args.type,
                task_index=args.task, config=c)

        X = get_placeholder(env.observation_space, name="states")
        A = tf.placeholder(tf.int32, [None], name="actions")
        R = tf.placeholder(tf.float32, [None], name="rewards")
        with tf.device(
                tf.train.replica_device_setter(worker_device=worker_device,
                    cluster=cluster)):
            with tf.variable_scope("global"):
                global_m = ICMModel(X, A, env)
            with tf.device(worker_device):
                with tf.variable_scope("local"):
                    local_m = ICMModel(X, A, env)
            expanded_a = tf.one_hot(A, env.action_space.n)
            lv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="local")
            gv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")
            log_logits = tf.nn.log_softmax(local_m.m.logits)
            pi_loss = - tf.reduce_sum(tf.stop_gradient (R - local_m.value) *
                    tf.reduce_sum(log_logits * expanded_a, axis=1))
            value_loss = tf.reduce_sum(tf.squared_difference(local_m.value, R))
            entropy = - tf.reduce_sum(log_logits * local_m.m.prob)
            bs = tf.cast(tf.shape(X)[0], tf.float32)
            tf.summary.scalar("model/policy_loss", pi_loss / bs)
            tf.summary.scalar("model/value_loss", value_loss / bs)
            tf.summary.scalar("model/entropy", entropy/ bs)
            tf.summary.scalar("model/v_norm", tf.global_norm(lv))

            # icm_f_err = tf.reduce_sum(local_m.icm.r_i)
            # log_logits = tf.nn.log_softmax(local_m.icm.pred_logits)
            # icm_b_err = tf.reduce_sum(-log_logits * expanded_a[:-1])
            # tf.summary.scalar("model/icm_b", icm_b_err / bs)
            # tf.summary.scalar("model/icm_f", icm_f_err / bs)
            # tf.summary.histogram("icm/rewards", local_m.icm.r_i)

            error = pi_loss + 0.25 * value_loss - 0.01 * entropy

            # beta = 0.2
            # error +=  beta * icm_f_err + (1 - beta) * icm_b_err
            tf.summary.scalar("model/error", error / bs)
            sync = [v1.assign(v2) for v1, v2 in zip(lv, gv)]
            g = tf.gradients(error, lv)
            g, _ = tf.clip_by_global_norm(g, 5e4)
            global_step = tf.Variable(0, trainable=False)
            gvs = list(zip(g, gv))
            train_op = tf.group(
                    tf.train.AdamOptimizer(1e-4).apply_gradients(gvs),
                    global_step.assign_add(tf.cast(bs, tf.int32)))
            tf.summary.scalar("model/gradient_norm", tf.global_norm(g))

        ready_op = tf.report_uninitialized_variables(gv)
        summary_op = tf.summary.merge_all()

        sw = tf.summary.FileWriter(os.path.join(args.logdir,
            "{}".format(args.task)))
        sv = tf.train.Supervisor(logdir=args.logdir,
                ready_op=ready_op,
                is_chief=(args.task==0),
                summary_op=None,
                summary_writer=None)

        config = tf.ConfigProto(inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1)
        with sv.managed_session(server.target, config=config) as sess:
            sw.add_graph(sess.graph)
            def policy(o, s):
                if s is None:
                    return sess.run([local_m.sample_action,
                        local_m.m.state],
                        feed_dict={local_m.input: o[None]})
                else:
                    return sess.run([local_m.sample_action,
                        local_m.m.state],
                        feed_dict={local_m.input: o[None],
                            local_m.m.initial_state: s})
            dgen = stateful_generator(env, policy)
            while not sv.should_stop():
                sess.run(sync)
                run = next(dgen)
                Re = 0
                R_ = []
                if run['t']:
                    gs = sess.run(global_step)
                    if 'tr' in run:
                        sw.add_summary(get_scalar_summary("game/score", run['tr']),
                                global_step=gs)
                    if 'len' in run:
                        sw.add_summary(get_scalar_summary("game/length", run['len']),
                                global_step=gs)
                    print("Finished run of length:", run['len'])
                else:
                    feed_dict = {local_m.input: run['s'][-1:]}
                    if run['ps'] is not None:
                        feed_dict.update({local_m.m.initial_state: run['ps']})
                    Re = sess.run(local_m.value, feed_dict=feed_dict).squeeze()
                ic_rewards = None
                feed_dict = {local_m.input: run['s'],
                        A: run['a']}
                if run['is'] is not None:
                    feed_dict.update({local_m.m.initial_state: run['is']})
                ic_rewards = sess.run(local_m.icm.r_i, feed_dict=feed_dict)
                discount = 0.99
                for r, ir in zip(run['r'][::-1], ic_rewards[::-1]):
                    Re = r + np.clip(0.00 * ir, -0.1, 0.1) + discount * Re
                    R_ = [Re] + R_
                feed_dict = {local_m.input: run['s'], A: run['a'], R: R_}
                if run['is'] is not None:
                    feed_dict.update({local_m.m.initial_state: run['is']})
                summaries, _ = sess.run([summary_op, train_op], feed_dict=feed_dict)
                if args.task == 0:
                    sw.add_summary(summaries, global_step=sess.run(global_step))

    else:
        raise ValueError("Type must be worker or ps, not {}".format(args.type))


def synchronous_a2c(env_str):
    pass

if __name__ == '__main__':
    distributed_icm()

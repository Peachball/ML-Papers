import tensorflow as tf
from utils import *
import numpy as np
import argparse
import sys
import gym
try:
    import gym_ple
except ImportError:
    pass

class PolicyValueModel():
    def __init__(self, inp, output_space, use_lstm=False):
        self.input = inp
        self.output_space = output_space

        act, val = self.build_net(self.input, use_lstm=use_lstm)
        self.action = act
        self.value = val

    def build_net(self, inp, use_lstm=False):
        k_init = None
        net = inp
        if len(inp.get_shape()) == 4:
            # Images
            net = tf.image.resize_images(net, [64, 64])
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


class OnlineTrainer():
    def __init__(self, model, env, sess, stateful=False):
        self.model = model
        self.env = env
        self.sess = sess
        self.stateful = stateful

    def get_run(self, max_len=10**10, render=False):
        states = []
        actions = []
        rewards = []
        m = self.model
        o = self.env.reset()
        d = False
        r = 0
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
        return {'s': states,
                'a': actions,
                'r': rewards,
                'terminal': d}


def _parse_distributed_a3c():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int)
    parser.add_argument('--cluster_file', type=str)
    parser.add_argument('--env', type=str, default='Pong-v0')
    parser.add_argument('--type', type=str)

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
    tf.summary.scalar("Entropy", entropy / tf.cast(tf.shape(A)[0],
        tf.float32))
    opt = tf.train.AdamOptimizer(1e-5)
    g = [grad
            for grad in tf.gradients(total_error, local_var)]
    g, global_norm = tf.clip_by_global_norm(g, 1e3)
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
            trainer = OnlineTrainer(local_model, env, sess, stateful=False)
            while not sv.should_stop():
                sess.run(sync)
                print("Getting run...")
                run = trainer.get_run()
                print("Played game of length {}".format(len(run['a'])))
                if run['terminal']:
                    R = 0
                else:
                    R = sess.run(local_model.value,
                            feed_dict={X: run['s'][-1]}).squeeze()
                Rewards = []
                lmd = 0.99
                for r in run['r'][::-1]:
                    R = lmd * R + r
                    Rewards = [R] + Rewards

                assert(len(run['s']) == len(run['a']) == len(Rewards))
                sw.add_summary(
                        get_scalar_summary("game/length", len(run['s'])),
                        sess.run(global_step))
                sw.add_summary(
                        get_scalar_summary("game/Score", sum(run['r'])),
                        sess.run(global_step))
                while len(Rewards) != 0:
                    sess.run(sync)
                    batch_size = 32
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


if __name__ == '__main__':
    distributed_a3c()

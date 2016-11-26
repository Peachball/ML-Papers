from __future__ import print_function
import tensorflow as tf
import gym
import gym_ple
import numpy as np

CONTINUOUS=False
ENV="Centipede-v0"
TRAINING=False
MAX_DATA_SIZE=10000
MAX_ASYNC_DATA_SIZE=100
MOVES = 1
GAMMA = 0.99

def conv_layer(X, shape, name, strides=[1,1,1,1], reuse=False):
    '''
        Shape: [nb filt, height, width]
    '''

    assert len(shape) == 3
    shape = [shape[1], shape[2], int(X.get_shape()[1]), shape[0]]
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        X = tf.transpose(X, perm=[0, 2, 3, 1])
        w = tf.get_variable('w', shape=shape, dtype=tf.float32)
        b = tf.get_variable('b', shape=[1, 1, 1, shape[-1]], dtype=tf.float32)
        out = tf.nn.conv2d(X, w, strides, 'SAME', data_format='NHWC') + b
        return tf.transpose(out, perm=[0, 3, 1, 2])

def act(inp, name):
    assert name in ['relu', 'softplus']
    if name=='relu':
        return tf.nn.relu(inp)
    if name=='softplus':
        return tf.nn.softplus(inp)

def pool(inp, dim=[2, 2]):
    inp = tf.transpose(inp, perm=[0, 2, 3, 1])
    inp = tf.nn.max_pool(inp, [1] + dim + [1], [1] + dim + [1], 'SAME',
            data_format='NHWC')
    return tf.transpose(inp, perm=[0, 3, 1, 2])

def build_conv_detector(image, reuse=False):
    ACTIVATION='softplus'

    with tf.variable_scope('ImageProcessor') as scope:
        net = conv_layer(image, [8, 7, 7], 'conv_1')
        net = act(net, 'softplus')
        net = pool(net, [4, 4])

        net = conv_layer(net, [32, 7, 7], 'conv_2')
        net = act(net, 'softplus')
        net = pool(net)

        net = conv_layer(net, [32, 7, 7], 'conv_3')
        net = act(net, 'softplus')
        net = pool(net)
    return net

def build_lstm_detector(reuse=False):
    cell = tf.nn.rnn_cell.LSTMCell(256, use_peepholes=True)
    lstm = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
    return lstm

def get_lstm_zero_state(batch_size=1):
    lstm = build_lstm_detector()
    return lstm.zero_state(batch_size, tf.float32)

def p_func(processed_s, batch_size=1, init_state=None):
    with tf.variable_scope('VPProcessor') as scope:
        dim_s = sum(processed_s.get_shape()[1:])

def _add_layer(x, out_dim, name):
    assert len(x.get_shape()) == 2
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [x.get_shape()[1], out_dim], tf.float32)
        b = tf.get_variable('b', [out_dim], tf.float32)
        return tf.matmul(x, w) + b

def predict_move(state, discrete_moves=1, continuous_moves=0, init_state=None,
        batch_size=1, reuse=False):
    assert continuous_moves == 0
    with tf.variable_scope("ValuePolicyLSTM", reuse=reuse) as scope:
        lstm = build_lstm_detector()
        if init_state is None:
            init_state = lstm.zero_state(batch_size, tf.float32)

        s = tf.expand_dims(state, 0)
        out, state = tf.nn.dynamic_rnn(lstm, s, initial_state=init_state)

    with tf.variable_scope("PolicyTransform", reuse=reuse) as scope:
        out = _add_layer(tf.squeeze(out, [0]), discrete_moves, "policy")
        out = tf.nn.softmax(out)

    return out, state

def vp_func(state,
        discrete_moves=1,
        continuous_moves=0,
        init_state=None,
        batch_size=1,
        reuse=False):
    assert continuous_moves == 0 # Not implemented yet
    assert len(state.get_shape()) == 2 # Only works with one time sequence

    states = tf.expand_dims(state, 0)
    with tf.variable_scope("ValuePolicyLSTM") as scope:
        lstm = build_lstm_detector()
        if init_state is None:
            init_state = lstm.zero_state(batch_size, tf.float32)

        outs, state = tf.nn.dynamic_rnn(lstm, states, initial_state=init_state,
                swap_memory=True)

    with tf.variable_scope("PolicyTransform") as scope:
        p = _add_layer(tf.squeeze(outs, [0]), discrete_moves, "policy")
        p = tf.nn.softmax(p)

    with tf.variable_scope("ValueTransform") as scope:
        v = _add_layer(tf.squeeze(outs, [0]), 1, "value")

    return p, v, state

def add_data(l, d):
    l.append(d)
    if len(d) > MAX_DATA_SIZE:
        del d[MAX_DATA_SIZE/10:]

def run_agent(env, sess, var, data, display=False):
    X = var['X']
    act = var['act']
    state = var['state']
    init_state = var['init_state']
    values = var['values']
    vp_train_op = var['vp_train_op']
    A_mask = var['A_mask']
    Adv = var['Adv']
    V_l = var['V_l']
    total_error = var['total_error']
    while TRAINING:
        done = False
        o = env.reset().transpose(2, 0, 1)[None,:,:,:]
        states = []

        train_init_state = lstm_state = None

        while not done:
            if display:
                env.render()
            prev_o = o
            if lstm_state is None:
                action, lstm_state = sess.run([act, state],
                        feed_dict={X: prev_o})
            else:
                action, lstm_state = sess.run([act, state],
                        feed_dict={X: prev_o, init_state: lstm_state})
            action = np.squeeze(action)
            action = np.random.choice(MOVES, 1, p=action)

            o, r, done, info = env.step(action)
            o = o.transpose(2, 0, 1)
            o = o[None,:,:,:]

            states.append((prev_o, action, r))
            add_data(data, (prev_o, action, r, o))

            if len(states) > MAX_ASYNC_DATA_SIZE:
                # Train using existing data
                R = sess.run(values, feed_dict={X: o})
                R = np.squeeze(R)
                sta, a, _= zip(*states)
                sta = np.array(sta).squeeze(1)
                a = np.array(a )
                R_t = []
                mask = np.zeros((a .shape[0], MOVES))
                mask[:,a] = 1
                for i, s in enumerate(states):
                    _, _, rew = s
                    R_t.insert(0, GAMMA * R + rew)
                    R = R_t[0]
                R_t = np.array(R_t)[:,None]
                # print("R", R_t.shape, "act", act.shape, "State", sta.shape)
                if train_init_state is not None:
                    adv = R_t - sess.run(values,
                            feed_dict={X: sta, init_state: train_init_state})
                else:
                    adv = R_t - sess.run(values,
                            feed_dict={X: sta})

                del states[:]
                train_init_state = lstm_state

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
            description='Train an agent to play an atari game')
    parser.add_argument('--env', dest='env',
            help='Specify open ai environment to play')

    args = parser.parse_args()
    ENV = args.env

def _product(l):
    a = 1
    for i in l:
        a = a * int(i)
    return a

def main():
    import time
    import threading
    parse_args()

    env = gym.make(ENV)
    global MOVES
    MOVES = (env.action_space.n)

    # Images
    X = tf.placeholder(tf.float32, [None, 3, 250, 160])

    # List of state sequences
    # Tuples of (state sequence, action sequence, reward sequence)
    data = tf.FIFOQueue(128, [tf.float32, tf.int32, tf.float32])
    V_l = tf.placeholder(tf.float32, [None, 1])
    A_mask = tf.placeholder(tf.float32, [None, MOVES])
    Adv = tf.placeholder(tf.float32, [None, 1])
    init_state = get_lstm_zero_state()

    features = build_conv_detector(X)
    print(features.get_shape())
    features = tf.reshape(features, [-1, _product(features.get_shape()[1:])])
    print(features.get_shape())

    actions, values, states = vp_func(features, discrete_moves=MOVES,
            init_state=init_state)
    act, state = predict_move(features, discrete_moves=MOVES,
            init_state=init_state, reuse=True)

    v_error = tf.reduce_sum(tf.squared_difference(values, V_l))
    p_error = tf.reduce_sum(
            -tf.log(tf.clip_by_value(A_mask * actions, 0.001, 0.999)) * Adv)

    total_error = v_error + p_error

    lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='ValuePolicyLSTM')
    p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='PolicyTransform')
    i_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='ImageProcessor')
    v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='ValueTransform')

    with tf.device('/gpu:0'):
        v_optimizer = tf.train.RMSPropOptimizer(1e-5)
        gvs = v_optimizer.compute_gradients(v_error,
                var_list=v_vars+i_vars+lstm_vars)

        # v_grads = [tf.Variable(tf.zeros_like(gv[1]), trainable=False) for gv in gvs]
        # accum_v_op = [v_grads[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        # reset_v_grad = tf.initialize_variables(v_grads)

        # v_train_op = v_optimizer.apply_gradients(
                # [(g.assign(g), gv[1]) for g, gv in zip(v_grads, gvs)])
        v_train_op = v_optimizer.apply_gradients(
                [(tf.clip_by_average_norm(gv[0], 5), gv[1]) for gv in gvs])

        p_opt = tf.train.RMSPropOptimizer(1e-5)
        gvs = p_opt.compute_gradients(p_error, var_list=i_vars+p_vars+lstm_vars)
        p_train_op = p_opt.apply_gradients(
                [(tf.clip_by_average_norm(gv[0], 5), gv[1]) for gv in gvs])

        vp_train_op = [p_train_op, v_train_op]

    init_op = tf.initialize_all_variables() #tf.initialize_variables(i_vars + lstm_vars + p_vars + v_vars)

    var = {
            'X': X, 'act': act, 'state': state, 'init_state': init_state,
            'values': values, 'vp_train_op': vp_train_op, 'A_mask': A_mask,
            'Adv': Adv, 'V_l': V_l, 'total_error': total_error
            }
    config = tf.ConfigProto(
            # device_count={'GPU': 0}
            )
    data = []
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        global TRAINING
        TRAINING = True
        threads = []

        envs = [gym.make(ENV) for i in range(1)]
        for i in envs:
            t = threading.Thread(target=run_agent,
                    args=(i, sess, var, data, True))
            t.start()
            threads.append(t)

            time.sleep(5) # 3 is arbitrary number, but it needs to pause for a
                          # sec here to let things render independently

        time.sleep(10)
        TRAINING = False

        for t in threads:
            t.join()


if __name__ == '__main__':
    main()

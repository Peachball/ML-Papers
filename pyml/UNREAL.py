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

def conv_layer(X, shape, name, strides=[1,1,1,1], reuse=False):
    '''
        Shape: [nb filt, height, width]
    '''

    assert len(shape) == 3
    shape = [shape[1], shape[2], int(X.get_shape()[1]), shape[0],]
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        w = tf.get_variable('w', shape=shape, dtype=tf.float32)
        b = tf.get_variable('b', shape=[shape[0]], dtype=tf.float32)
        out = tf.nn.conv2d(X, w, strides, 'SAME', data_format='NCHW')
        return out

def act(inp, name):
    assert name in ['relu', 'softplus']
    if name=='relu':
        return tf.nn.relu(inp)
    if name=='softplus':
        return tf.nn.softplus(inp)

def pool(inp, dim=[2, 2]):
    return tf.nn.max_pool(inp, [1,1] + dim, [1,1] + dim, 'SAME',
            data_format='NCHW')

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

        out, state = lstm(state, init_state)

    with tf.variable_scope("PolicyTransform", reuse=reuse) as scope:
        out = _add_layer(out, discrete_moves, "policy")
        out = tf.nn.softmax(out)

    return out, state

def vp_func(state, discrete_moves=1, continuous_moves=0, init_state=None,
        batch_size=1, reuse=False):
    assert continuous_moves == 0
    with tf.variable_scope("ValuePolicyLSTM") as scope:
        lstm = build_lstm_detector()
        if init_state is None:
            init_state = lstm.zero_state(batch_size, tf.float32)

        out, state = lstm(state, init_state)

    with tf.variable_scope("PolicyTransform") as scope:
        p = _add_layer(out, discrete_moves, "policy")
        p = tf.nn.softmax(p)

    with tf.variable_scope("ValueTransform") as scope:
        v = _add_layer(out, 1, "value")

    return p, v, state

def add_data(l, d):
    l.append(d)
    if len(d) > MAX_DATA_SIZE:
        del d[MAX_DATA_SIZE/10:]

def run_agent(env, sess, X, act, data, state, init_state, display=False):
    while TRAINING:
        done = False
        o = env.reset()
        states = []

        lstm_state = None

        while not done:
            if display:
                env.render()
            o = o.transpose(2, 0, 1)
            prev_o = o[None,:,:,:]
            if lstm_state is None:
                action, lstm_state = sess.run([act, state],
                        feed_dict={X: prev_o})
            else:
                action, lstm_state = sess.run([act, state],
                        feed_dict={X: prev_o, init_state: lstm_state})
            action = np.squeeze(action)
            action = np.random.choice(MOVES, 1, p=action)

            o, r, done, info = env.step(action)
            states.append((prev_o, action))
            add_data(data, (prev_o, action, r, o))

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
    V_u = tf.placeholder(tf.float32, [None, 1])
    init_state = get_lstm_zero_state()

    features = build_conv_detector(X)
    features = tf.reshape(features, [-1, _product(features.get_shape()[1:])])

    values, actions, states = vp_func(features, discrete_moves=MOVES,
            init_state=init_state)
    act, state = predict_move(features, discrete_moves=MOVES,
            init_state=init_state, reuse=True)

    init_op = tf.initialize_all_variables()

    data = []
    with tf.Session() as sess:
        sess.run(init_op)
        global TRAINING
        TRAINING = True
        threads = []

        envs = [gym.make(ENV) for i in range(8)]
        for i in envs:
            t = threading.Thread(target=run_agent,
                    args=(i, sess, X, act, data, state, init_state, True))
            t.start()
            threads.append(t)

            time.sleep(4) # 3 is arbitrary number, but it needs to pause for a
                          # sec here

        time.sleep(10)
        TRAINING = False

        for t in threads:
            t.join()


if __name__ == '__main__':
    main()

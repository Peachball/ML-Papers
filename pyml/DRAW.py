import theano
import theano.tensor as T
import numpy as np
import tensorflow as tf

def load_mnist():
    from keras.datasets import mnist
    return mnist.load_data()


def add_fc_layer(name, inp, out_size):
    shape = list(inp.get_shape())
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', [shape[-1], out_size])
        b = tf.get_variable('b', [out_size])

        if len(shape) == 3:
            reduced = tf.reshape(inp, [-1, shape[-1]])
            multiplied = tf.matmul(inp, w) + b
            return mulitiplied
        if len(shape) == 2:
            return tf.matmul(inp, w) + b
        raise Exception("Not implemented")

def DRAWNetwork():
    (X_t, _), (X_v, _) = load_mnist()

    X = tf.placeholder(tf.float32, [None, 784], name='Input')
    batch_size = 32
    timesteps = 10
    latent_size = 300

    x_bat = tf.placeholder(tf.float32, [batch_size, 784], name='batch_inp')

    with tf.variable_scope("encoder") as scope:
        lstm = tf.nn.rnn_cell.LSTMCell(500, use_peepholes=True)
        enc_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 3)
        enc_state = enc_lstm.zero_state(batch_size, tf.float32)
        e_state = enc_state

    with tf.variable_scope("decoder") as scope:
        lstm = tf.nn.rnn_cell.LSTMCell(500, use_peepholes=True)
        dec_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 3)
        dec_state = dec_lstm.zero_state(batch_size, tf.float32)
        d_state = dec_state

    init_canvas = tf.constant(0.0, dtype=tf.float32, shape=[batch_size, 784], name='canvas')
    init_h_dec = tf.constant(0.0, tf.float32, [batch_size, 784], name='decoded')
    init_h_enc = tf.constant(0.0, tf.float32, [batch_size, 300], name='encoded')

    canvas = init_canvas
    h_dec = init_h_dec
    h_enc = init_h_enc

    for i in range(timesteps):
        x_hat = x_bat - tf.sigmoid(canvas)
        r_t = tf.concat(1, [X, x_hat, h_dec])
        with tf.variable_scope("encoder") as scope:
            output, e_state = enc_lstm(r_t, e_state)
            scope.reuse_variables()

        vals = add_fc_layer('fc', output, 2 * latent_size)
        z_mean, z_log_var = tf.split(1, 2, vals)
        print(z_mean.get_shape())

        z = tf.random_normal(z_mean.get_shape()) * tf.exp(z_log_var / 2.0)\
                + z_mean

        with tf.variable_scope("decoder") as scope:
            h_dec = dec_lstm(tf.concat(1, [z, h_dec]), d_state)
            scope.reuse_variables()
        canvas = canvas + h_dec

        tf.get_variable_scope().reuse_variables()


    final_canvas = canvas
    final_state = [e_state, d_state]

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)

if __name__ == '__main__':
    DRAWNetwork()

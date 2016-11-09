import theano
import theano.tensor as T
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_mnist():
    from keras.datasets import mnist
    (a, _), (c, _) = mnist.load_data()
    a = a.astype('float32') / 255.0
    c = c.astype('float32') / 255.0
    a = a.reshape(-1, 784)
    c = c.reshape(-1, 784)
    return a, c

def add_fc_layer(name, inp, out_size, share=True):
    shape = list(inp.get_shape())
    with tf.variable_scope(name) as scope:
        if share:
            scope.reuse_variables()
        w = tf.get_variable('w', [shape[-1], out_size])
        b = tf.get_variable('b', [out_size])

        if len(shape) == 3:
            reduced = tf.reshape(inp, [-1, shape[-1]])
            multiplied = tf.matmul(inp, w) + b
            return mulitiplied
        if len(shape) == 2:
            return tf.matmul(inp, w) + b
        raise Exception("Not implemented")

def read(x_t, x_hat, dec_h):
    return tf.concat(1, [x_t, x_hat])

def _get_encoder(hidden_size=256):
    lstm = tf.nn.rnn_cell.LSTMCell(hidden_size, use_peepholes=True)
    return tf.nn.rnn_cell.MultiRNNCell([lstm] * 1)

def encode(enc_h, r, dec_h, hidden_size=256, init_state=None, batch_size=32,
        share=True):
    with tf.variable_scope("encoder") as scope:
        if share:
            scope.reuse_variables()
        encoder = _get_encoder(hidden_size)
        if init_state is None:
            init_state = encoder.zero_state(batch_size, tf.float32)

        h_enc, state = encoder(tf.concat(1, [r, dec_h]), init_state)

    return h_enc, state

def _get_encoder_state(hidden_size=256, batch_size=32):
    with tf.variable_scope("encoder") as scope:
        encoder = _get_encoder(hidden_size)
        init_state = encoder.zero_state(batch_size, tf.float32)
        return init_state

def _get_decoder(hidden_size=256):
    lstm = tf.nn.rnn_cell.LSTMCell(hidden_size, use_peepholes=True)
    return tf.nn.rnn_cell.MultiRNNCell([lstm] * 1)

def decode(dec_h, z, hidden_size=256, init_state=None, batch_size=32, share=True):
    with tf.variable_scope("decoder") as scope:
        if share:
            scope.reuse_variables()
        decoder = _get_decoder(hidden_size)
        if init_state is None:
            init_state = decoder.zero_state(batch_size, tf.float32)

        dec_h, state = decoder(z, init_state)
    return dec_h, state

def _get_decoder_state(hidden_size=256, batch_size=32):
    with tf.variable_scope("decoder") as scope:
        decoder = _get_decoder()
        init_state = decoder.zero_state(batch_size, tf.float32)
        return init_state

def sample(enc_h, latent_dim=300, share=True, latent_loss=None):
    l = add_fc_layer("sampler", enc_h, latent_dim * 2, share=share)
    z_mean, z_log_var = tf.split(1, 2, l)
    z = tf.random_normal(z_mean.get_shape()) * tf.exp(z_log_var / 2.0) +\
            z_mean
    if latent_loss is not None:
        latent_loss += 0.5 * tf.reduce_mean(tf.square(z_mean) +
                tf.exp(z_log_var) -
                z_log_var + 1)
    return z, latent_loss

def write(dec_h, out_size=784, share=True):
    l = add_fc_layer("write", dec_h, out_size, share=share)
    return l

def DRAWNetwork():
    batch_size = 32
    timesteps = 10
    latent_size = 10
    logdir = 'tflogs/draw/'

    X_t, X_v = load_mnist()
    from tensorflow.examples.tutorials import mnist
    X_b = mnist.input_data.read_data_sets('/tmp/mnist', one_hot=True).train

    X = tf.placeholder(tf.float32, [batch_size, 784], name='Input')

    init_canvas = canvas = tf.constant(0.0, tf.float32, [batch_size, 784])
    init_h_enc_state = h_enc_state = _get_encoder_state()
    init_h_dec_state = h_dec_state = _get_decoder_state()
    init_h_enc = h_enc = tf.constant(0.0, tf.float32, [batch_size, 256])
    init_h_dec = h_dec = tf.constant(0.0, tf.float32, [batch_size, 256])

    share = False
    drawings = [canvas]
    latent_loss = 0.0

    for i in range(timesteps):
        x_hat = X - tf.sigmoid(canvas)
        r_t = read(X, x_hat, h_dec)
        h_enc, h_enc_state = encode(h_enc, r_t, h_dec, init_state=h_enc_state,
                share=share)
        z, latent_loss = sample(h_enc, share=share, latent_loss=latent_loss,
                latent_dim=latent_size)
        h_dec, h_dec_state = decode(h_dec, z, init_state=h_dec_state,
                share=share)
        canvas = canvas + write(h_dec, share=share)
        drawings += [tf.sigmoid(canvas)]
        share = True

    final_canvas = canvas
    final_enc_state = h_enc
    final_dec_state = h_dec

    Lx = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(tf.sigmoid(canvas),
        X), 1))

    loss = Lx + latent_loss
    tf.scalar_summary("Loss", loss)

    summaries = tf.merge_all_summaries()
    saver = tf.train.Saver()
    opt = tf.train.AdamOptimizer(0.001, beta1=0.5)
    grad = opt.compute_gradients(loss)
    for i, (g, v) in enumerate(grad):
        if g is not None:
            grad[i] = (tf.clip_by_norm(g, 5), v)

    train_step = opt.apply_gradients(grad)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        plt.ion()
        sw = tf.train.SummaryWriter(logdir, sess.graph)
        sess.run(init_op)
        iteration = 0
        for i in range(1000):
            print(X_b.next_batch(batch_size)[0].shape)
            s, _, e, d, lat = sess.run([summaries, train_step, loss, drawings,
                latent_loss],
                    feed_dict={X: X_b.next_batch(batch_size)[0]})
            print("Lx: {} Latent: {}".format(e, lat))
            sw.add_summary(s)
            iteration += 1
            if iteration % 100 == 0:
                saver.save(sess, logdir + 'model', global_step=iteration)

            plt.subplot(121)
            plt.imshow(d[10][0].reshape(28, 28), cmap='Greys')
            plt.subplot(122)
            plt.imshow(X_t[i].reshape(28, 28), cmap='Greys')
            plt.pause(0.05)

if __name__ == '__main__':
    DRAWNetwork()

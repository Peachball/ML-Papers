# import theano
# import theano.tensor as T
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

def data_gen(batch_size=32,
        img_size=[1024, 1024],
        data_dir='/home/peachball/D/git/ML-Papers/datasets/hehexdDataSet'):
    from PIL import Image
    from os.path import join
    from os import listdir
    files = listdir(data_dir)
    i = 0
    while True:
        d = []
        while len(d) < 32:
            try:
                img = Image.open(join(data_dir, files[i]))
                img = img.convert('RGB')
                img = img.resize(tuple(img_size))
                d.append(np.array(img))
            except:
                print('Unable to open file {}'.format(files[i]))
            finally:
                i += 1
                i = i % len(files)
        d = np.array(d).astype('float32') / 255.0
        d = d.transpose(0, 3, 1, 2)
        yield d

def _shape(x):
    return x.get_shape()

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
        w = tf.get_variable('w', [shape[-1], out_size],
                initializer=tf.random_uniform_initializer(-0.005, 0.005))
        b = tf.get_variable('b', [out_size],
                initializer=tf.random_uniform_initializer(-0.005, 0.005))

        if len(shape) == 3:
            reduced = tf.reshape(inp, [-1, shape[-1]])
            multiplied = tf.matmul(inp, w) + b
            return mulitiplied
        if len(shape) == 2:
            return tf.matmul(inp, w) + b
        raise Exception("Not implemented")

def _get_filter(gx, gy, mux, muy, sigma, xsize, ysize, N):
    sigma = tf.expand_dims(sigma, 1)
    sigma = tf.expand_dims(sigma, 1)
    sigma = tf.check_numerics(sigma, 'sigma problem')

    mux = tf.expand_dims(mux, 2)
    Fx = tf.cast(tf.expand_dims(tf.range(xsize), 0), tf.float32)
    Fx = tf.tile(Fx, [N, 1])
    Fx = tf.expand_dims(Fx, 0)
    Fx = tf.exp(-tf.square(Fx - mux) / (2 * sigma))
    Fx = tf.check_numerics(Fx, 'Fx before partition')
    Z = (tf.expand_dims(tf.reduce_sum(Fx, 1), 1))
    Fx = Fx / (tf.abs(Z) + 1e-8)

    muy = tf.expand_dims(muy, 2)
    Fy = tf.cast(tf.expand_dims(tf.range(ysize), 0), tf.float32)
    Fy = tf.tile(Fy, [N, 1])
    Fy = tf.expand_dims(Fy, 0)
    Fy = tf.exp(-tf.square(Fy - muy) / (2 * sigma))
    Z = (tf.expand_dims(tf.reduce_sum(Fy, 1), 1))
    Fy = Fy / (tf.abs(Z) + 1e-8)

    return (Fx, Fy)


def test_filter():
    x = tf.placeholder(tf.float32, [28, 28])
    img, _ = load_mnist()
    img = img[:32]
    N = 10
    mux = tf.cast(tf.range(N), tf.float32)
    muy = tf.cast(tf.range(N), tf.float32)
    delta = (28 - 1) / (N - 1)
    mux = 14 + (mux  - N/2 - 0.5) * delta
    muy = 14 + (muy  - N/2 - 0.5) * delta
    Fx, Fy = _get_filter(14, 14, mux, muy, 1.0, 28, 28, N)
    out = tf.matmul(tf.matmul(Fy, x), tf.transpose(Fx))
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        filtered = sess.run(out, feed_dict={x: img[0].reshape(28, 28)})
    plt.imshow(filtered, cmap='Greys')

    plt.figure()
    plt.imshow(img[0].reshape(28, 28), cmap='Greys')
    plt.show()

def read(x_t, x_hat, dec_h):
    return tf.concat(1, [x_t, x_hat])

def _get_encoder(hidden_size=256):
    lstm = tf.nn.rnn_cell.LSTMCell(hidden_size, use_peepholes=True)
    return tf.nn.rnn_cell.MultiRNNCell([lstm] * 3)

def encode(enc_h, r, dec_h, hidden_size=256, init_state=None, batch_size=32,
        share=True):
    with tf.variable_scope('encoder') as scope:
        if share:
            scope.reuse_variables()
        encoder = _get_encoder(hidden_size, share=share)
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
    return tf.nn.rnn_cell.MultiRNNCell([lstm] * 3)

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
    timesteps = 20
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

    latent_loss = latent_loss / timesteps
    final_canvas = tf.sigmoid(canvas)
    final_enc_state = h_enc
    final_dec_state = h_dec

    Lx = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(final_canvas, X), 1))

    loss = Lx + latent_loss
    tf.summary.scalar("Loss", loss)

    summaries = tf.summary.merge_all()
    saver = tf.train.Saver()
    lr = tf.Variable(0.001)
    opt = tf.train.AdamOptimizer(lr, beta1=0.5)
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
        rate = 0.001
        ema = 1000
        p_ema = ema
        for i in range(10000):
            x_d = X_b.next_batch(batch_size)[0]
            s, _, e, d, lat = sess.run([summaries, train_step, loss, drawings,
                latent_loss],
                feed_dict={X: x_d, lr:rate})
            print("Lx: {} Latent: {}".format(e, lat))
            sw.add_summary(s)
            iteration += 1
            if iteration % 100 == 0:
                saver.save(sess, logdir + 'model', global_step=iteration)

            ema = (0.99 * ema) + 0.01 * e
            if iteration % 10 == 0:
                print('EMA: {}'.format(ema))
                if p_ema < ema:
                    rate /= 2.0
                    print('Learning rate decreased')
                p_ema = ema
            if iteration % 20 == 0:
                img = np.zeros((10 * 28, 20 * 28)).astype('float32')
                for i in range(10):
                    for j in range(20):
                        img[i*28:(i+1)*28, j*28:(j+1)*28] = d[j][i].reshape(28, 28)
                    img[i*28:(i+1)*28, :28] = x_d[i].reshape(28, 28)
                plt.imshow(img, cmap='Greys')
                plt.pause(0.05)

def _generate_mu(g, delta, length=10):
    mu = tf.cast(tf.range(length), tf.float32)
    mu = tf.expand_dims(mu, 0)
    g = tf.expand_dims(g, 1)
    delta = tf.expand_dims(delta, 1)
    return g + (mu - length / 2  - 0.5) * delta

def read_attn(x_t, x_hat, dec_h, N=5, share=True):
    combined = tf.concat(1, [x_t, x_hat])
    channels = tf.split(1, 2 * int(x_t.get_shape()[1]), combined)
    output = []
    batch_size = x_t.get_shape()[0]
    img_size = x_t.get_shape()[2:]

    # Get params (mus, sigmas, deltas, gammas)
    params = add_fc_layer('attention', dec_h, 5, share=share)
    gx_ = (int(img_size[0]) + 1) / 2 * (tf.tanh(params[:,0]) + 1)
    gy_ = (int(img_size[1]) + 1) / 2 * (tf.tanh(params[:,1]) + 1)
    delta = (tf.cast(tf.maximum(img_size[0], img_size[1]) - 1, tf.float32))\
                / (N - 1) * tf.exp(params[:,2])
    log_sig = params[:,3]
    log_gamma = params[:,4]

    sigma = tf.exp(log_sig)
    gamma = tf.exp(log_gamma)

    mux = _generate_mu(gx_, delta, length=N)
    muy = _generate_mu(gy_, delta, length=N)

    img_x_size = (x_t.get_shape()[2])
    img_y_size = (x_t.get_shape()[3])
    Fx, Fy = _get_filter(gx_, gy_, mux, muy, sigma, img_x_size, img_y_size, N)

    read = []
    for m in channels:
        my = tf.batch_matmul(Fy, tf.squeeze(m))
        myx = tf.batch_matmul(my, tf.transpose(Fx, perm=[0, 2, 1]))
        b_gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        read.append(b_gamma * myx)
    return tf.transpose(tf.pack(read), perm=[1, 0, 2, 3])

def encode_attn(h_enc, r_t, h_dec, share=True, hidden_size=256, init_state=None):
    with tf.variable_scope('attn_encoder') as scope:
        if share:
            scope.reuse_variables()
        encoder = _get_encoder(hidden_size)
        if init_state is None:
            init_state = encoder.zero_state()
        inp = tf.concat(1, [h_enc, r_t, h_dec])
        h_enc, state = encoder(inp, init_state)

        return h_enc, state

def write_attn(dec_h, N=5, channels=3, share=True, image_size=None, timestep=0):
    assert image_size is not None
    img_size = image_size
    batch_size = int(dec_h.get_shape()[0])
    w_t = add_fc_layer('region', dec_h, N*N*channels, share=share)
    w_p = tf.reshape(w_t, [-1, channels, N, N])
    split = tf.split(1, channels, w_p)

    params = add_fc_layer('attention', dec_h, 5, share=True)
    # tf.histogram_summary("params_" + str(timestep), params)
    gx_ = (int(img_size[0]) + 1) / 2 * (tf.tanh(params[:,0]) + 1)
    gy_ = (int(img_size[1]) + 1) / 2 * (tf.tanh(params[:,1]) + 1)
    delta = (tf.cast(tf.maximum(img_size[0], img_size[1]) - 1, tf.float32))\
    / (N - 1) * tf.exp(params[:,2])
    log_sig = params[:,3]
    log_gamma = params[:,4]

    sigma = tf.exp(log_sig)
    gamma = tf.exp(log_gamma)

    # tf.histogram_summary("sigma_dist_" + str(timestep), sigma)
    # tf.histogram_summary("gamma_dist_" + str(timestep), gamma)
    # tf.histogram_summary("gx_dist_" + str(timestep), gx_)
    # tf.histogram_summary("gy_dist_" + str(timestep), gy_)
    # tf.histogram_summary("delta_dist_" + str(timestep), delta)

    mux = _generate_mu(gx_, delta, length=N)
    muy = _generate_mu(gy_, delta, length=N)

    img_x_size = img_size[0]
    img_y_size = img_size[1]
    Fx, Fy = _get_filter(gx_, gy_, mux, muy, sigma, img_x_size, img_y_size, N)

    write = []
    for m in split:
        my = tf.batch_matmul(tf.transpose(Fy, perm=[0, 2, 1]), tf.squeeze(m))
        myx = tf.batch_matmul(my, Fx)
        b_gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        write.append(myx / (b_gamma + 0.001))
    return tf.transpose(tf.pack(write), perm=[1, 0, 2, 3])

def mnist_gen(batch_size=32):
    from keras.datasets import mnist
    (a, _), (c, _) = mnist.load_data()
    a = a.astype('float32') / 255.0
    c = c.astype('float32') / 255.0
    a = a.reshape(-1, 1, 28, 28)
    c = c.reshape(-1, 1, 28, 28)
    d = np.concatenate((a, c), axis=0)
    while True:
        for i in range(0, d.shape[0], 32):
            if i + 32 > d.shape[0]:
                continue
            yield d[i:i+32]

def BigDraw(demo=False):
    batch_size=32
    timesteps = 30
    latent_size= 100
    decoded_size = 500
    encoded_size = 500
    N = 10
    image_size = [1, 28, 28]
    SAVE_PATH = 'tflogs/mnist_draw_attn'

    X = tf.placeholder(tf.float32, [batch_size] + image_size)

    init_dec_h = dec_h = tf.constant(0, shape=[batch_size, decoded_size],
            dtype=tf.float32)
    init_enc_h = enc_h = tf.constant(0, shape=[batch_size, encoded_size],
            dtype=tf.float32)
    init_states = enc_state = dec_state = _get_encoder_state(hidden_size=500)
    init_canvas = canvas = tf.constant(0, shape=X.get_shape(), dtype=tf.float32)

    share = False
    latent_loss = 0.0
    drawings = []
    for i in range(timesteps):
        print('\rFinished layer {} out of {}'.format(i+1, timesteps), end="")
        x_hat = X - tf.sigmoid(canvas)
        r_t = read_attn(X, x_hat, dec_h, share=share, N=N)
        r_t = tf.reshape(r_t, [batch_size, -1])
        enc_h, enc_state = encode_attn(enc_h, r_t, dec_h, share=share,
                hidden_size=500,
                init_state=enc_state)

        z, latent_loss = sample(enc_h, latent_dim=latent_size, share=share,
                latent_loss=latent_loss)
        dec_h, dec_state = decode(dec_h, z, hidden_size=500,
                init_state=dec_state, batch_size=batch_size, share=share)

        c_t = write_attn(dec_h, share=share,
                image_size=image_size[1:], timestep=i, channels=image_size[0])
        canvas = canvas + c_t
        drawings.append(tf.sigmoid(canvas))
        share=True
    print()
    final_canvas = canvas
    latent_loss = tf.reduce_sum(latent_loss / timesteps)
    Lx = tf.squared_difference(tf.sigmoid(final_canvas), X)
    Lx = tf.reduce_sum(Lx, [2, 3])
    Lx = tf.reduce_mean(Lx)

    loss = Lx + latent_loss
    tf.summary.scalar('loss', loss)

    summaries = tf.summary.merge_all()
    lr = tf.Variable(1e-3)
    global_step = tf.Variable(0)
    opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-3)
    grad = opt.compute_gradients(loss)
    for i, (g, v) in enumerate(grad):
        if g is not None:
            grad[i] = (tf.clip_by_norm(g, 5), v)

    train_step = opt.apply_gradients(grad, global_step=global_step)

    writer = tf.summary.FileWriter(SAVE_PATH)
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    drawings = np.zeros((batch_size, timesteps) + image_size)
    def trainer():
        ema = 1000
        p_e = ema
        with tf.Session() as sess:
            try:
                print('Attempting to load previous graph...')
                cpkt = tf.train.get_checkpoint_state(SAVE_PATH)
                saver.restore(sess, cpkt.model_checkpoint_path)
            except Exception as e:
                print(e)
                print("Unable to load previous graph")
                sess.run(init_op)
            writer.add_graph(sess.graph)
            alpha = lr.eval()
            print('Current learning rate: {}'.format(alpha))

            print('Beginning generation')
            for d in mnist_gen(batch_size=batch_size):
                (_,
                 e,
                 s,
                 dr,
                 lx,
                 lz,
                 i
                  ) = sess.run([train_step,
                      loss,
                      summaries,
                      drawings,
                      Lx,
                      latent_loss,
                      global_step], feed_dict={X: d, lr: alpha})
                writer.add_summary(s, global_step=i)
                ema = 0.99 * ema + 0.01 * e
                if i % 20 == 0 and ema > p_e:
                    print("Learning rate decreased")
                    alpha = alpha / 2
                if i % 10 == 0:
                    drawings = dr
                    # plt.subplot(121)
                    # plt.imshow(np.squeeze(dr[-1][0].transpose(1, 2, 0)),
                            # cmap='Greys')
                    # plt.subplot(122)
                    # plt.imshow(d[0].transpose(1, 2, 0).squeeze(), cmap='Greys')
                    # plt.pause(0.05)
                if i % 50 == 0:
                    saver.save(sess, SAVE_PATH, ep=global_step)
                print(e, lx, lz, ema)

    t = threading.Thread(target=trainer())
    t.start()

    def show_image(i):
        return drawings[]
        pass

    def update(*args):
        pass

    fig = plt.figure()


if __name__ == '__main__':
    BigDraw()

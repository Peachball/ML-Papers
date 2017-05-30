#!/usr/bin/env python3
import tensorflow as tf
from datautils import *
from utils import *
import matplotlib.pyplot as plt
import threading
import time
import os
import queue
import random


class DRAGAN():
    def __init__(self, inp):
        self.input = inp
        self.Z = tf.random_normal([tf.shape(inp)[0], 100])
        self.build_net()


    def build_net(self):
        self.generated = self.generator(self.Z)

        self.real_disc = self.discriminator(self.input)
        self.fake_disc = self.discriminator(self.generated, reuse=True)
        curscope = tf.get_variable_scope().name
        if len(curscope) > 0:
            curscope = curscope + "/"
        self.gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=curscope + "generator")
        self.disc_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=curscope + "discriminator")


    def generator(self, inp, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.layers.dense(inp, 1024, name="projection")
            net = tf.nn.elu(net)
            net = tf.reshape(net, [tf.shape(inp)[0], 2, 2, 256])
            net = tf.layers.conv2d_transpose(net, 1024, [3, 3], strides=(2, 2),
                    name="deconv0", padding="same")
            net = tf.nn.elu(net)
            net = tf.layers.conv2d_transpose(net, 512, [5, 5], strides=(2, 2),
                    padding="same", name="deconv1")
            net = tf.nn.elu(net)
            net = tf.layers.conv2d_transpose(net, 256, [5, 5], strides=(2, 2),
                    padding="same", name="deconv2")
            net = tf.nn.elu(net)
            net = tf.layers.conv2d_transpose(net, 128, [5, 5], strides=(2, 2),
                    padding="same", name="deconv3")
            net = tf.nn.elu(net)
            net = tf.layers.conv2d_transpose(net, 4, [5, 5], strides=(2, 2),
                    padding="same", name="deconv4")
            net = tf.nn.sigmoid(net)
            return net


    def discriminator(self, inp, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            net = inp
            for i in range(4):
                net = tf.layers.conv2d(net, 64 * 2 ** i, [5, 5], strides=(2, 2),
                        name="conv{}".format(i + 1))
                net = tf.nn.elu(net)
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 1, name="fc")
            return net


class DRAGANTrainer():
    def __init__(self, inp, m):
        lmbda = 1e1
        self.m = m
        self.input = inp
        mean, var = tf.nn.moments(inp, axes=[0, 1, 2, 3])
        tf.summary.scalar("inp_stddev", tf.sqrt(var), collections=['train'])
        delta = 0.5 * tf.sqrt(var) * tf.random_uniform(
                tf.shape(inp), minval=-1.0, maxval=1.0)
        alpha = tf.random_uniform([tf.shape(inp)[0], 1, 1, 1])
        perturbed = inp * alpha + (1 - alpha) * (inp + delta)
        tf.summary.image("perturbed_image", perturbed, collections=['train'])
        with tf.variable_scope("dragan"):
            avg_p = tf.reduce_mean(m.discriminator(perturbed, reuse=True))
        grad_n = tf.global_norm(tf.gradients(avg_p, perturbed))
        disc_error = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(m.real_disc),
                    logits=m.real_disc)
                + tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(m.real_disc),
                    logits=m.fake_disc))
        disc_penalty = tf.reduce_mean(tf.square(grad_n - 1.0))
        total_d_error = disc_error + lmbda * disc_penalty
        gen_error = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(m.fake_disc),
                    logits=m.fake_disc))
        tf.summary.scalar('disc_cross_entropy', disc_error,
                collections=['train'])
        tf.summary.scalar("discriminator_penalty", disc_penalty,
                collections=['train'])
        tf.summary.scalar("discriminator_error", total_d_error,
                collections=["train"])
        tf.summary.scalar("generator_error", gen_error, collections=["train"])
        global_step = tf.Variable(0, trainable=False, name="global_step")
        disc_train_op = tf.train.RMSPropOptimizer(1e-5).minimize(total_d_error,
                global_step=global_step, var_list=m.disc_var)
        gen_train_op = tf.train.RMSPropOptimizer(1e-5).minimize(gen_error,
                global_step=global_step, var_list=m.gen_var)

        self.global_step = global_step
        self.d_train_op = disc_train_op
        self.g_train_op = gen_train_op
        self.d_err = total_d_error
        self.g_err = gen_error


class GAN:
    def __init__(self, image_dim, generator=None, discriminator=None,
            batch_size=32):
        if generator is None:
            generator = self.build_generator
        self.generator = generator
        if discriminator is None:
            discriminator = self.build_discriminator
        self.discriminator = discriminator
        self.image_dim = image_dim
        self.queue = queue
        self.session_linked = False

        self.data_queue = tf.FIFOQueue(10000, [tf.float32], [image_dim])
        self.training_data = self.data_queue.dequeue_many(batch_size)

        self.image_p = tf.placeholder(tf.float32, [None] + image_dim)

        self.enqueue_op = self.data_queue.enqueue_many([self.image_p])


    def link_session(self, sess):
        self.sess = sess
        self.session_linked = True


    def add_to_queue(self, data):
        assert self.session_linked
        self.sess(self.enqueue_op, feed_dict={self.image_p: data})


    def build_generator(self, Z):
        s = Z.get_shape()
        assert len(s) == 2
        with tf.variable_scope('generator'):
            net = tf.reshape(Z, [-1, 2, 2, 25])
            net = add_conv_layer(net, [3, 3, 64])


    def build_discriminator(self, X):
        pass


def load_image(index, dataset='hehexdDataSet', image_dim=(64, 64), rgba=False):
    from os.path import join, isfile
    from os import listdir
    dataset = join('..', 'datasets', dataset)
    files = [f for f in listdir(dataset) if isfile(join(dataset, f))]
    if index >= len(files):
        index = index % len(files)

    from PIL import Image
    img = Image.open(join(dataset, files[index]))
    if rgba:
        img = img.convert('RGBA')
    else:
        img = img.convert('RGB')
    img = img.resize(image_dim)
    return np.array(img)


def data_gen(batch_size=32, channels_first=True, rgba=False):
    index = 0
    while True:
        im = []
        for i in range(batch_size):
            im.append(load_image(index, rgba=rgba))
            index += 1
        im = np.array(im).astype('float32') / 255.0
        if channels_first:
            im = im.transpose(0, 3, 1, 2)
        yield im


def load_images(r, dataset='hehexdDataSet', image_dim=[224,224]):
    if not isinstance(r, tuple):
        raise Exception("IDK how to read that non tuple bro")
    if not len(r) == 2:
        raise Exception("incorrect tuple length")

    imgs = []
    for i in range(r[0], r[1]):
        imgs.append(load_image(i, dataset=dataset, image_dim=image_dim))

    return np.array(imgs).astype('float32') / 255


def generate(z=None, batch_size=None):
    assert (z is None) ^ (batch_size is None)

    if z is None:
        z = tf.random_normal([batch_size, 128])
    if batch_size is None:
        batch_size = -1
    features = tf.reshape(z, [batch_size, 2, 2, 32])
    w_init = tf.random_uniform_initializer(-0.005, 0.005)

    with tf.variable_scope("upsample1"):
        net = add_deconv_layer(features, [3, 3], 64, [2, 2], 'tconv',
                w_init=w_init)
        net = tf.nn.elu(net)
        net = add_conv_layer(net, [3, 3], 64, "conv", w_init=w_init)
        net = tf.nn.elu(net)

    with tf.variable_scope("upsample2"):
        net = add_deconv_layer(net, [3, 3], 96, [2, 2], 'tconv', w_init=w_init)
        net = tf.nn.elu(net)
        net = add_conv_layer(net, [3, 3], 96, "conv", w_init=w_init)
        net = tf.nn.elu(net)

    for i in range(3):
        with tf.variable_scope("upsample{}".format(i+3)):
            net = add_deconv_layer(net, [3, 3], 128, [2, 2],
                    'tconv'.format(i+3), w_init=w_init)
            net = tf.nn.elu(net)
            net = add_conv_layer(net, [3, 3], 128, "conv", w_init=w_init)
            net = tf.nn.elu(net)

    net = add_conv_layer(net, [3, 3], 3, "conv7", w_init=w_init)

    net = tf.sigmoid(net)

    return net


def discriminate(X):
    with tf.variable_scope("conv1"):
        net = add_conv_layer(X, [3, 3], 64, "conv")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

    with tf.variable_scope("conv2"):
        net = add_conv_layer(net, [3, 3], 96, "conv")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

    with tf.variable_scope("conv3"):
        net = add_conv_layer(net, [3, 3], 96, "conv")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

    with tf.variable_scope("conv4"):
        net = add_conv_layer(net, [3, 3], 128, "conv")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

    with tf.variable_scope("conv5"):
        net = add_conv_layer(net, [3, 3], 128, "conv")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

    net = flatten(net)
    net = add_layer(net, 1, "fc")

    return net


def data_loader(sess, enqueue, X, coord, image_dim):
    i = 0
    while not coord.should_stop():
        images = load_images((i, i+32), image_dim=image_dim)
        images = images.astype('float32')
        i += 32
        sess.run(enqueue, feed_dict={X: images})


def trainer(sess, g_train, d_train, coord, summaries, writer, global_step):
    while not coord.should_stop():
        for i in range(5):
            _, s = sess.run([d_train, summaries])
            writer.add_summary(s, global_step=sess.run(global_step))
        _, s = sess.run([g_train, summaries])
        writer.flush()


def train_GAN(dataset=None):
    IMAGE_SIZE = [64, 64]
    BATCH_SIZE = 32
    LOGDIR = 'tflogs/hehedcgan'
    if dataset is None:
        dataset = 'hehexdDataSet'
    x = load_images((0, 12), dataset, image_dim=IMAGE_SIZE)

    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, [None] + IMAGE_SIZE + [3], name='Images')
        Z = tf.placeholder(tf.float32, [1, 128], name='noise')
        data_queue = tf.RandomShuffleQueue(1024, 128, [tf.float32], [64, 64, 3])
        enqueue_op = data_queue.enqueue_many((X,))
        q_image = data_queue.dequeue_many(32)
        tf.summary.image("Training images", q_image)

    with tf.variable_scope("generator") as scope:
        output = generate(batch_size=32)
        tf.summary.image("Generated images", output)
        scope.reuse_variables()
        custom_output = generate(z=Z)

    with tf.variable_scope("discriminator") as scope:
        d = discriminate(output)
        scope.reuse_variables()
        real = discriminate(q_image)

    with tf.name_scope("error"):
        d_error = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(d, tf.zeros_like(d)))
        tf.summary.scalar("Fake discriminator error", d_error)

        dr_error = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(real, tf.ones_like(real)))
        tf.summary.scalar("Real discriminator error", dr_error)

        g_error = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(d, tf.ones_like(d)))
        tf.summary.scalar("Generator error", g_error)

    with tf.name_scope('gradients'):
        global_step = tf.Variable(0, trainable=False)

        g_train_opt = tf.train.AdamOptimizer(0.001)
        gvs = g_train_opt.compute_gradients(
                g_error,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope="generator"))
        capped_gvs = [(tf.clip_by_norm(g, 5.0), v) for g, v in gvs]
        g_train_op = g_train_opt.apply_gradients(capped_gvs,
                global_step=global_step)

        d_train_opt = tf.train.GradientDescentOptimizer(0.001)
        d_train_opt.compute_gradients(
                d_error + dr_error,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope='discriminator'))
        capped_gvs = [(tf.clip_by_norm(g, 5.0), v) for g, v in gvs]
        d_train_op = d_train_opt.apply_gradients(capped_gvs,
                global_step=global_step)
        iterations = tf.Variable(0)

        train_op = tf.group(g_train_op, d_train_op)

    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    coord = tf.train.Coordinator()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(LOGDIR)
        if ckpt is not None:
            saver.restore(sess, ckpt)
        else:
            sess.run(init_op)
            writer.add_graph(sess.graph)

        t = [
                threading.Thread(target=trainer, args=(sess, g_train_op,
                    d_train_op, coord, summaries, writer, global_step)),
                threading.Thread(target=data_loader, args=(sess, enqueue_op,
                    X, coord, IMAGE_SIZE)),
            ]

        for i in t:
            i.start()

        while True:
            try:
                generated_images = sess.run(custom_output,
                        feed_dict={Z: np.random.normal(size=(1, 128))})
                plt.subplot(121)
                plt.imshow(generated_images[0])

                generated_images = sess.run(custom_output,
                        feed_dict={Z: np.random.normal(size=(1, 128))})
                plt.subplot(122)
                plt.imshow(generated_images[0])
                plt.show()

                saver.save(sess, os.path.join(LOGDIR, 'model'), global_step=global_step)
            except KeyboardInterrupt:
                print("\rStopping threads")
                coord.request_stop()
                coord.join(t)
                exit()


def train_DRAGAN(LOGDIR="tflogs/dragan"):
    X = tf.placeholder(tf.float32, [None, 64, 64, 4])
    tf.summary.image("input_image", X, collections=['train'])
    with tf.variable_scope("dragan"):
        m = DRAGAN(X)
    tf.summary.image("generated_image", m.generated, collections=["train"])

    with tf.name_scope("trainer"):
        t = DRAGANTrainer(X, m)

    summary_op = tf.summary.merge_all(key="train")
    sw = tf.summary.FileWriter(LOGDIR)
    sv = tf.train.Supervisor(logdir=LOGDIR,
            summary_op=None,
            summary_writer=None)
    dq = queue.Queue(5)
    def get_batches(q, batch_size=32, img_dir="../datasets/hehexd_64x64/images"):
        from PIL import Image
        ls = os.listdir(img_dir)
        i = 0
        while True:
            batch = np.zeros((batch_size, 64, 64, 4), "float32")
            for j in range(batch_size):
                try:
                    img = Image.open(os.path.join(img_dir, ls[i%len(ls)]))
                    img = img.convert('RGBA')
                    img_arr = np.array(img)
                    batch[i%32] = img_arr
                    i += 1
                except KeyboardInterrupt:
                    exit()
                except:
                    pass
            q.put(batch / 255.0)

    dl = threading.Thread(target=get_batches, args=(dq,))
    dl.start()

    with sv.managed_session() as sess:
        sw.add_graph(sess.graph)
        data = dq.get()
        while not sv.should_stop():
            # d_err = 10
            # while d_err > 1:
                # s, _, d_err = sess.run([summary_op, t.d_train_op, t.d_err],
                        # feed_dict={m.input: data})
                # sw.add_summary(s, global_step=sess.run(t.global_step))
                # data = dq.get()
            # g_err = 10
            # while g_err > 1:
                # s, _, g_err = sess.run([summary_op, t.g_train_op, t.g_err],
                        # feed_dict={m.input: data})
                # sw.add_summary(s, global_step=sess.run(t.global_step))
            s, _, _ = sess.run([summary_op, t.g_train_op, t.d_train_op],
                    feed_dict={m.input: data})
            data = dq.get()
            sw.add_summary(s, global_step=sess.run(t.global_step))


if __name__ == '__main__':
    train_DRAGAN()

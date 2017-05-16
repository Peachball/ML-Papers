"""
Implementations of the various pixel architextures e.g. pixelcnn and pixelrnn
from: https://arxiv.org/pdf/1601.06759.pdf
"""
import tensorflow as tf
from datautils import *
import matplotlib.pyplot as plt
import threading
import os


class PixelCNN():
    def __init__(self, image_dim, batch_size=None):
        self.x = tf.placeholder(tf.int32, [batch_size] + image_dim)
        self.summaries = []
        self.logits = self._build_net(tf.cast(self.x, tf.float32) / 255.0)
        self.output = tf.nn.softmax(self.logits)

    def _build_net(self, inp):
        net = inp
        net = self._add_conv_layer(net, 128, [7, 7], name='conv1', mask='A')
        net = tf.nn.elu(net)

        for i in range(15):
            with tf.variable_scope("residual_block_{}".format(i)):
                downsampled = tf.layers.conv2d(net, 64, [1, 1])
                ds = tf.nn.elu(downsampled)

                ds = self._add_conv_layer(
                        ds, 64, [5, 5], name='masked_conv', mask='B')
                ds = tf.nn.elu(ds)

                upsampled = tf.layers.conv2d(ds, 128, [1, 1])
                net = tf.nn.elu(upsampled) + net

        # Convert to 256 softmax
        net = tf.layers.conv2d(net, 256, [1, 1])

        # Hack
        net = tf.expand_dims(net, axis=-2)
        return net

    def _add_conv_layer(self, inp, filters, shape, name='conv', mask='A'):
        assert(shape[0] % 2 == 1 and shape[1] % 2 == 1)
        with tf.variable_scope(name):
            ins = inp.get_shape().as_list()
            fs = list(shape) + [ins[-1], filters]
            w = tf.get_variable('w', shape=fs)
            b = tf.get_variable('b', shape=[1, 1, 1, filters])
            msk = np.ones(fs).astype('float32')
            xcen = shape[0] // 2
            ycen = shape[1] // 2
            msk[xcen:,ycen:,:,:] = 0
            msk[xcen+1:,:,:,:] = 0
            if mask == 'B':
                msk[xcen,ycen,:,:] = 1
            msk = tf.constant(msk)
            w = w * msk
            return tf.nn.convolution(inp, w, 'SAME') + b

    def sample(self, n, sess):
        canvas = np.zeros((n, n, 1))
        for r in range(n):
            for c in range(n):
                canvas[r,c,0] = np.random.choice(
                        256,
                        p=sess.run(self.output,
                            feed_dict={self.x: canvas[None]})[0,r,c,0])
        return canvas


def test_pixelcnn(logdir='tflogs/naive_pixelcnn'):
    with tf.name_scope("pixelcnn"):
        m = PixelCNN([28, 28, 1])
    im_one_hot = tf.one_hot(m.x, 256)
    im_prob = tf.reduce_prod(tf.reduce_sum(m.output * im_one_hot, axis=-1), axis=[1, 2])
    log_error = tf.losses.softmax_cross_entropy(im_one_hot, m.logits)
    tf.add_to_collection("train", tf.summary.scalar("Error", log_error))

    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("gradients"):
        opt = tf.train.AdamOptimizer(1e-2, epsilon=0.1)
        gvs = opt.compute_gradients(log_error)
        clipped_gvs = [(tf.clip_by_norm(g, 5), v) for g, v in gvs]
        train_op = opt.apply_gradients(clipped_gvs, global_step=global_step)

    sw = tf.summary.FileWriter(logdir)
    saver = tf.train.Saver()

    gen_image = tf.placeholder(tf.float32, [None, 28, 28, 1])
    tf.add_to_collection("misc", tf.summary.image("Generated_image",
        gen_image))

    train_summaries = tf.summary.merge(tf.get_collection("train"))
    misc_summaries = tf.summary.merge(tf.get_collection("misc"))
    with tf.Session() as sess:
        cpkt = tf.train.latest_checkpoint(logdir)
        try:
            saver.restore(sess, cpkt)
        except:
            sess.run(tf.global_variables_initializer())
            sw.add_graph(sess.graph)
            print("Unable to load previous checkpoint")
        X, _ = load_mnist()
        plt.ion()
        ind = 0
        for e in range(30):
            for i in range(0, len(X), 32):
                print("Training")
                if ind % 100 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, e, s = sess.run([train_op, log_error, train_summaries],
                            feed_dict={m.x: X[i:i+32,:,:,None]},
                            options=run_options,
                            run_metadata=run_metadata)
                    sw.add_run_metadata(run_metadata,
                            "step%d" % sess.run(global_step),
                            global_step=sess.run(global_step))
                else:
                    _, e, s = sess.run([train_op, log_error, train_summaries],
                            feed_dict={m.x: X[i:i+32,:,:,None]})
                print("Error at batch {}: {}".format(i, e))
                sw.add_summary(s, global_step=sess.run(global_step))
                if ind % 100 == 0:
                    saver.save(sess, os.path.join(logdir, 'model'),
                            global_step=sess.run(global_step))
                    [summary] = sess.run([misc_summaries],
                            feed_dict={gen_image: m.sample(28, sess)[None]})
                    sw.add_summary(summary, global_step=sess.run(global_step))
                ind += 1


if __name__ == '__main__':
    test_pixelcnn()

"""
Implementations of the various pixel architextures e.g. pixelcnn and pixelrnn
from: https://arxiv.org/pdf/1601.06759.pdf
"""
import tensorflow as tf
from datautils import *


class PixelCNN():
    def __init__(self, image_dim, batch_size=None):
        self.x = tf.placeholder(tf.int32, [batch_size] + image_dim)
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
                        ds, 64, [3, 3], name='masked_conv', mask='B')
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
            mask = np.ones(fs).astype('float32')
            xcen = shape[0] // 2
            ycen = shape[1] // 2
            mask[xcen:,ycen:,:,:] = 0
            mask[xcen+1:,:,:,:] = 0
            if mask == 'B':
                mask[xcen,ycen,:,:] = 1
            msk = tf.constant(mask)
            w = w * msk
            return tf.nn.convolution(inp, w, 'SAME') + b


def test_pixelcnn():
    m = PixelCNN([28, 28, 1])
    im_one_hot = tf.one_hot(m.x, 256)
    print(im_one_hot.get_shape())
    log_error = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(im_one_hot, m.logits))
    train_op = tf.train.GradientDescentOptimizer(1e-2).minimize(log_error)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        X, _ = load_mnist()
        for i in range(0, len(X), 32):
            _, e = sess.run([train_op, log_error], feed_dict={m.x:
                X[i:i+32,:,:,None]})
            print("Error at batch {}: {}".format(i, e))



if __name__ == '__main__':
    test_pixelcnn()

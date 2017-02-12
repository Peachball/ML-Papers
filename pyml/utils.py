import tensorflow as tf

def add_layer(l, out_dim, name, reuse=False, w_init=None, b_init=None,
        collections=None):
    '''
        out_dim is a scalar
    '''
    with tf.variable_scope(name, reuse=reuse):
        l_s = l.get_shape()
        w = tf.get_variable("w", shape=[l_s[1], out_dim], initializer=w_init)
        b = tf.get_variable("bias", shape=[out_dim], initializer=b_init)
        if collections:
            tf.add_to_collection(collection, w)
            tf.add_to_collection(collection, b)
        return tf.matmul(l, w) + b


def add_conv_layer(l, filt_dim, out_channels, name, strides=[1, 1], reuse=None, w_init=None,
        b_init=None):
    '''
        filt_dim is of the shape [x, y, output channels]
    '''
    assert len(l.get_shape()) >= 4
    assert len(filt_dim) == 2
    with tf.variable_scope(name, reuse=reuse) as scope:
        im_s = l.get_shape()
        w_s = filt_dim + [im_s[-1]] + [out_channels]
        w = tf.get_variable("filter", shape=w_s, initializer=w_init)
        b_s = [1, 1, 1, out_channels]
        b = tf.get_variable("bias", shape=b_s, initializer=b_init)
        return tf.nn.conv2d(l, w, [1] + strides + [1], "SAME") + b


def max_pool(l, pdim):
    s = [1] + pdim + [1]
    return tf.nn.max_pool(l, s, s, 'SAME')


def flatten(l):
    dim = 1
    shape = l.get_shape().as_list()
    for i in shape[1:]:
        dim *= i
    return tf.reshape(l, [-1, dim])


def add_deconv_layer(l, filt_dim, out_channels, strides, name, reuse=None,
        w_init=None, b_init=None):
    assert len(filt_dim) == 2
    assert len(l.get_shape()) == 4
    with tf.variable_scope(name, reuse=reuse) as scope:
        i_s = tf.shape(l)
        static_s = l.get_shape().as_list()

        batch_size = static_s[0]

        out_shape = tf.pack([batch_size, strides[0] * static_s[1], strides[1] *
            static_s[2], out_channels])
        w_s = filt_dim + [out_channels] + [l.get_shape()[-1]]
        w = tf.get_variable("filt", shape=w_s, initializer=w_init)
        b = tf.get_variable("bias", shape=[1, 1, 1, out_channels],
                initializer=b_init)
        return tf.nn.conv2d_transpose(l, w, out_shape, [1] + strides + [1]) + b

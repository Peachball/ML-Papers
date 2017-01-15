import tensorflow as tf

def add_layer(l, out_dim, name, reuse=False, w_init=None, b_init=None,
        collections=None):
    with tf.variable_scope(name, reuse=reuse):
        l_s = l.get_shape()
        w = tf.get_variable("w", shape=[l_s[1], out_dim], initializer=w_init)
        b = tf.get_variable("b", shape=[out_dim], initializer=b_init)
        if collections:
            tf.add_to_collection(collection, w)
            tf.add_to_collection(collection, b)
        return tf.matmul(l, w) + b


def add_conv_layer(l, filt_dim, name, strides=[1, 1], reuse=False, w_init=None,
        b_init=None):
    assert len(l.get_shape()) >= 4
    assert len(filt_dim) == 3
    with tf.variable_scope(name, reuse=reuse) as scope:
        im_s = l.get_shape()
        w_s = filt_dim[:2] + [im_s[-1]] + [filt_dim[-1]]
        w = tf.get_variable("filter", shape=w_s, initializer=w_init)
        b_s = [1, 1, 1, filt_dim[-1]]
        b = tf.get_variable("b", shape=b_s, initializer=b_init)
        return tf.nn.conv2d(l, w, [1] + strides + [1], "SAME") + b


def max_pool(l, pdim):
    s = [1] + pdim + [1]
    return tf.nn.max_pool(l, s, s, 'SAME')


def flatten(l):
    dim = 1
    for i in l.get_shape()[1:]:
        dim *= int(i)
    return tf.reshape(l, [-1, dim])


def openimages_generator():
    pass

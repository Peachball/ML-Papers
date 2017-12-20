import tensorflow as tf
import gym

def add_layer(l, out_dim, name, reuse=False, w_init=None, b_init=None,
        collections=None, biases=True):
    '''
        out_dim is a scalar
    '''
    with tf.variable_scope(name, reuse=reuse):
        l_s = l.get_shape()
        w = tf.get_variable("w", shape=[l_s[1], out_dim], initializer=w_init)
        if biases:
            b = tf.get_variable("bias", shape=[out_dim], initializer=b_init)
        if collections:
            tf.add_to_collection(collection, w)
            tf.add_to_collection(collection, b)
        if biases:
            return tf.matmul(l, w) + b
        else:
            return tf.matmul(l, w)


def add_layer_batch(l, out_dim, name, reuse=False, w_init=None, b_init=None):
    raise NotImplementedError()
    with tf.variable_scope(name, reuse=reuse):
        s_t = tf.shape(l)
        a_s = l.get_shape()


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
    """
        Note: for images only
    """
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


def clip_gvs(gvs, magnitude, clipping_func=tf.clip_by_norm):
    return [(clipping_func(g, magnitude), v) for (g, v) in gvs if g is not None]


def get_placeholder(space, name=None):
    if type(space) == gym.spaces.Box:
        return tf.placeholder(tf.float32, [None] + list(space.shape), name=name)
    if type(space) == gym.spaces.Discrete:
        return tf.placeholder(tf.int32, [None, space.n], name=name)
    # if type(space) == gym.spaces.Discrete:
        # return tf.placeholder(tf.int32, [None])
    raise NotImplementedError("Cannot handle type: {}".format(type(space)))


def get_scalar_summary(tag, value):
    s = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    return s


def run_env(env, sess, X, a, render=False, initial_observation=None):
    if initial_observation is None:
        o = env.reset()
    else:
        o = initial_observation
    d = False
    obs = []
    act = []
    rew = []
    info = []
    while not d:
        obs.append(o)
        if render:
            env.render()
        action = sess.run(a, feed_dict={X: o[None]})
        o, r, d, i = env.step(action)
        rew.append(r)
        act.append(action)
        info.append(i)
    return (obs, act, rew, info)



def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

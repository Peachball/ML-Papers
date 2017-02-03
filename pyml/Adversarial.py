import tensorflow as tf
import argparse
from utils import *
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.datasets import cifar10
import subprocess
import numpy as np


def build_model(X, batch_size=None, reuse=False):
    with tf.variable_scope('image_classifier', reuse=reuse) as scope:
        net = add_conv_layer(X, [5, 5, 32], "conv1")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

        net = add_conv_layer(net, [5, 5, 64], "conv2")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

        net = flatten(net)
        net = add_layer(net, 10, 'fc3')

    return net


def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate and visualize some adversarial images')

    parser.add_argument("--logdir", help="Tensorflow output directory",
            default="tflogs/advopenimages/", type=str)

    parser.add_argument("--model", help="Model you want to load (optional)",
            default=None)

    return parser.parse_args()


def generate_adversarial_example(image, grad, epsilon=0.1):
    sign = tf.sign(grad) * epsilon
    return image + sign


def main():
    args = parse_args()

    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
    tf.summary.image("Input data", x)
    y = tf.placeholder(tf.int32, [None], name='labels')
    out = build_model(x)
    prediction = tf.nn.softmax(out, name='prediction')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    ex_error = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(out, tf.one_hot(y, 10)),
            name='error')

    adv_ex = generate_adversarial_example(x, tf.gradients(ex_error, [x])[0],
            epsilon=0.25)
    adv_ex = tf.stop_gradient(adv_ex)

    adv_prediction = build_model(adv_ex, reuse=True)
    adv_error = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                adv_prediction,
                tf.one_hot(y, 10)),
            name='adv_error')

    error = 0 * ex_error + 1 * adv_error
    tf.summary.scalar("Error", error)
    total_correct = tf.Variable(0, name='total_correct')
    total_ex = tf.Variable(0, name='total_example')

    accum_accuracy = tf.group(
            tf.assign_add(total_correct, tf.reduce_sum(
                tf.cast(tf.equal(tf.cast(tf.argmax(out, axis=1), tf.int32), y),
                    tf.int32))),
            tf.assign_add(total_ex, tf.shape(y)[0]))
    total_accuracy = tf.truediv(total_correct, total_ex, name="total_accuracy")
    reset_accuracy = tf.group(
            tf.assign(total_correct, 0),
            tf.assign(total_ex, 0)
            )
    accuracy = tf.truediv(tf.reduce_sum(
            tf.cast(tf.equal(tf.cast(tf.argmax(out, axis=1), tf.int32), y), tf.int32)
            ), tf.shape(y)[0], name='accuracy')
    tf.summary.scalar("Accuracy", accuracy)

    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.logdir + 'train/')
    val_writer = tf.summary.FileWriter(args.logdir + 'validation/')
    with tf.name_scope('trainer'):
        train_op = tf.train.AdamOptimizer().minimize(error, global_step=global_step)
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()


    (X, Y), (X_v, Y_v) = mnist.load_data()
    Y = Y.squeeze()
    Y_v = Y_v.squeeze()

    X = X.astype('float32')[:,:,:,None] / 255.0
    X_v = X_v.astype('float32')[:,:,:,None] / 255.0

    with tf.Session() as sess:
        cpkt = tf.train.latest_checkpoint(args.logdir)
        print(cpkt)
        if args.model is not None:
            saver.restore(sess, args.model)
        elif cpkt is not None:
            saver.restore(sess, cpkt)
        else:
            sess.run(init_op)
            print("Unable to load any model")

        writer.add_graph(sess.graph, global_step=global_step.eval())

        plt.ion()
        while True:
            for i in range(0, 100, 32):
                [s, e, _, _] = sess.run([summaries, error, train_op, accum_accuracy],
                        feed_dict={x: X[i:i+32], y: Y[i:i+32]})
                writer.add_summary(s, global_step.eval())
                print("Error: {}".format(e))

                if global_step.eval() % 50 == 0:
                    saver.save(sess, args.logdir + 'model', global_step=global_step)
                    sess.run(reset_accuracy)
                    s = sess.run(summaries,
                            feed_dict={x: X_v[:1000], y: Y_v[:1000]})
                    val_writer.add_summary(s, global_step.eval())
                    val_writer.flush()

                    p = sess.run(prediction, feed_dict={x: X[i:i+1]}).squeeze()
                    plt.clf()
                    plt.subplot(221)
                    plt.imshow(X[i].squeeze(), cmap='Greys',
                            interpolation='none')
                    plt.annotate('Prediction: ' + str(p.argmax()), xy=(0, 0),
                            xytext=(0, -3))
                    plt.annotate('Confidence: ' + str(p[p.argmax()]), xy=(0, 0),
                            xytext=(0, -1))

                    ex = sess.run(adv_ex, feed_dict={x: X[i:i+1], y: Y[i:i+1]})
                    p = sess.run(prediction, feed_dict={x: ex}).squeeze()
                    plt.subplot(222)
                    plt.imshow(ex[0].squeeze(), cmap='Greys',
                            interpolation='none')
                    plt.annotate('Prediction: ' + str(p.argmax()), xy=(0, 0),
                            xytext=(0, -3))
                    plt.annotate('Confidence: ' + str(p[p.argmax()]), xy=(0, 0),
                            xytext=(0, -1))

                    plt.pause(0.05)


if __name__ == '__main__':
    main()

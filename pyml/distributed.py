import tensorflow as tf
from utils import *
import argparse
import openimages
import threading
import time
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="openimages", type=str,
            help="The project you want to run [openimages]")

    parser.add_argument("--logdir", default='tflogs/openimages/',
            type=str, help="Location to store tensorflow logs")

    args = parser.parse_args()
    return args


def build_model(X, reuse=False):
    with tf.variable_scope('image_classifier', reuse=reuse) as scope:
        net = add_conv_layer(X, [5, 5, 32], "conv1")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

        net = add_conv_layer(net, [5, 5, 64], "conv2")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

        net = add_conv_layer(net, [5, 5, 64], "conv3")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

        net = add_conv_layer(net, [5, 5, 64], "conv4")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

        net = add_conv_layer(net, [5, 5, 64], "conv5")
        net = max_pool(net, [2, 2])
        net = tf.nn.elu(net)

        net = flatten(net)
        net = add_layer(net, 7881, 'fc3')

    return net


def add_data_target(sess, enqueue_op, i_p, l_p, coord, start, stride,
        batch_size=8, image_size=(224, 224)):
    for i in openimages.next_images_batch((batch_size), size=image_size, seek=start,
            stride=stride):
        sess.run(enqueue_op, feed_dict={i_p: i[0], l_p: i[1]})
        if coord.should_stop():
            print("Stopping...")
            return


def train_target(sess, train_op, error, summaries, writer, global_step):
    while True:
        [_, e, s] = sess.run([train_op, error, summaries])
        writer.add_summary(s, global_step=sess.run(global_step))
        print("Error: {}".format(e))


def deep_dist_obj_detection(args):
    IMAGE_SIZE = (224, 224)
    batch_size = 32
    image_p = tf.placeholder(tf.float32, [None] + list(IMAGE_SIZE) +
        [3])
    label_p = tf.placeholder(tf.float32, [None, 7881])
    data_queue = tf.RandomShuffleQueue(2048, 64,
            [tf.float32, tf.float32],
            shapes=[list(IMAGE_SIZE) + [3], 7881])
    enqueue_op = data_queue.enqueue_many((image_p, label_p))

    (images, labels) = data_queue.dequeue_many(batch_size)
    tf.summary.image("Input images", images)
    logits = build_model(images)
    error = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits, labels), axis=1),
            name='error')
    tf.summary.scalar("Error", error)

    prediction = tf.sigmoid(logits)
    pred_ind = tf.concat(1,
            [tf.expand_dims(tf.range(prediction.get_shape()[0]), axis=1),
                tf.expand_dims(tf.cast(tf.argmax(prediction, axis=1), tf.int32), axis=1)])
    confidence = tf.sparse_to_dense(pred_ind, prediction.get_shape(), 1.0) * labels
    avg_confidence = tf.reduce_mean(tf.reduce_sum(confidence, axis=1))
    tf.summary.scalar("Average confidence", avg_confidence)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('gradients'):
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        gvs = opt.compute_gradients(error)
        capped_gvs = []
        grad_norm = tf.global_norm(list(zip(*gvs))[0])
        tf.summary.scalar("Gradient norm", grad_norm)
        for g, v in gvs:
            capped_gvs += [(tf.clip_by_norm(g, 5), v)]
        train_op = opt.apply_gradients(capped_gvs, global_step=global_step)

    summaries = tf.summary.merge_all()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    coord = tf.train.Coordinator()
    writer = tf.summary.FileWriter(args.logdir)

    with tf.Session() as sess:
        cpkt = tf.train.latest_checkpoint(args.logdir)
        try:
            saver.restore(sess, cpkt)
        except:
            sess.run(init_op)
            print("Unable to load previous checkpoint")

        writer.add_graph(sess.graph, global_step=global_step.eval())

        threads = []
        num_threads = 16
        load_size = 1
        for i in range(num_threads):
            t = threading.Thread(target=add_data_target,
                    args=(sess,
                        enqueue_op,
                        image_p,
                        label_p,
                        coord,
                        load_size * i,
                        num_threads * load_size,
                        load_size,
                        IMAGE_SIZE))
            t.start()
            print("Starting thread")
            threads.append(t)

        train_thread = threading.Thread(target=train_target,
                args=(sess, train_op, error, summaries, writer, global_step))
        train_thread.start()

        while True:
            try:
                time.sleep(120)
                saver.save(sess, os.path.join(args.logdir, 'model'), global_step=global_step)
            except KeyboardInterrupt:
                print("Stopping threads")
                coord.request_stop()
                coord.join(threads)
                exit()


if __name__ == '__main__':
    args = get_args()
    if args.project == 'openimages':
        deep_dist_obj_detection(args)

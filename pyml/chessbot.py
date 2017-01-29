from __future__ import print_function
from utils import *
import chess
import argparse
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", default='tflogs/chess/v4',
            type=str, help="Location to store tensorflow logs")

    parser.add_argument("--batch_size", default=32, type=int,
            help="Batch size to use when training")

    args = parser.parse_args()
    return args


def convert_to_array(board):
    P_T_I = {'k' : 1,
             'q' : 2,
             'p' : 3,
             'b' : 4,
             'r' : 5,
             'n' : 6}

    epd = board.epd().split(' ')[0]
    white = np.zeros((8, 8))
    black = np.zeros((8, 8))

    r = 7
    c = 0
    for p in epd:
        if p == '/':
            r -= 1
            c = 0
            continue

        try:
            s = int(p)
            c += s
            continue
        except ValueError:
            pass

        if p.istitle():
            white[c][r] = P_T_I[p.lower()]
            c += 1
            continue
        else:
            black[c][r] = P_T_I[p]
            c += 1
            continue

    return (white, np.rot90(black, k=2))


def play_self(sess, enqueue, X, Y, V, coord, verbose=False):
    b = chess.Board()
    b.reset()
    white_states = []
    black_states = []
    prev_winner = True
    while not coord.should_stop():
        if b.is_game_over():
            if b.is_stalemate():
                print("stalemate")
            else:
                r = b.result()
                if r[0] == '1':
                    ww = 1
                    bw = 0
                else:
                    ww = 0
                    bw = 1
                print("Played game of length {}, {} won"
                        .format(len(white_states), ww))
                sess.run(enqueue, feed_dict={X: np.array(white_states),
                    Y: [ww] * len(white_states)})
                sess.run(enqueue, feed_dict={X: np.array(black_states), Y:[bw] *
                    len(black_states)})

            del white_states[:]
            del black_states[:]
            b.reset()

        def select_move(white=True, T=-1):
            moves = []
            for m in b.legal_moves:
                moves.append(m)

            move_states = np.zeros((len(moves), 8, 8))
            i = 0
            for m in moves:
                b.push(m)
                whi, blk = convert_to_array(b)
                if white:
                    b_array = whi
                else:
                    b_array = blk
                move_states[i] = b_array
                i += 1
                b.pop()
            w_moves = sess.run(V, feed_dict={X: move_states}).squeeze(axis=1)
            if verbose: print(w_moves)

            if T is not None:
                if T > 0:
                    w_moves = np.exp(w_moves / T)
                    w_moves = w_moves / w_moves.sum()
                    m_i = np.random.choice(len(moves), p=w_moves)
                else:
                    m_i = np.random.choice(len(moves))
            else:
                m_i = np.argmax(w_moves)

            return moves[m_i]

        if b.turn:
            curmove = select_move(b.turn, -1)
            white, _ = convert_to_array(b)
            white_states.append(white)
        else:
            curmove = select_move(b.turn, None)
            _, black = convert_to_array(b)
            black_states.append(black)

        b.push(curmove)


def train(sess, train, coord, summaries, writer, global_step):
    while not coord.should_stop():
        _, s = sess.run([train, summaries])
        writer.add_summary(s, sess.run(global_step))
        writer.flush()


def build_net(X, histograms=False):
    net = add_conv_layer(X, [32, 3, 3], 'conv1')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_1', net)

    net = add_conv_layer(net, [64, 3, 3], 'conv2')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_2', net)

    net = add_conv_layer(net, [128, 3, 3], 'conv3')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_3', net)

    net = add_conv_layer(net, [128, 3, 3], 'conv4')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_4', net)

    net = add_conv_layer(net, [128, 3, 3], 'conv5')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_5', net)

    net = add_conv_layer(net, [128, 3, 3], 'conv6')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_6', net)

    net = add_conv_layer(net, [32, 3, 3], 'conv7')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_7', net)

    net = flatten(net)
    net = add_layer(net, 1, 'fc')
    return net


def main():
    args = get_args()
    data_queue = tf.RandomShuffleQueue(10000, 128, [tf.int64, tf.float32],
            [[8, 8], 1])
    X = tf.placeholder(tf.int64, [None, 8, 8], name='input')
    L_d = tf.placeholder(tf.float32, [None], name='win_status')
    enqueue_op = data_queue.enqueue_many([X, tf.expand_dims(L_d, axis=1)])
    B, L = data_queue.dequeue_many(args.batch_size)

    converted_board = tf.one_hot(B, 7)

    with tf.variable_scope("valuenet") as scope:
        value = build_net(converted_board)

    with tf.variable_scope("valuenet", reuse=True):
        quick_v = build_net(tf.one_hot(X, 7))

    error = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(value, L))
    tf.summary.scalar("Error", error)

    accuracy = tf.reduce_mean(tf.cast(tf.greater(tf.sign(value) * tf.sign(L - 0.5), 0),
            tf.float32))
    tf.summary.scalar("Accuracy", accuracy)

    global_step = tf.Variable(0, trainable=False)

    coord = tf.train.Coordinator()
    with tf.name_scope("gradients"):
        opt = tf.train.AdamOptimizer(0.001)
        gvs = opt.compute_gradients(error)
        clipped_gvs = [(tf.clip_by_norm(g, 5.0), v) for g, v in gvs]
        train_op = opt.apply_gradients(clipped_gvs, global_step=global_step)

    init_op = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.logdir)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(args.logdir)
        try:
            saver.restore(sess, ckpt)
        except:
            sess.run(init_op)
            print("Unable to load previous checkpoint")
            writer.add_graph(sess.graph, global_step=global_step.eval())

        t = [threading.Thread(target=play_self,
            args=(sess, enqueue_op, X, L_d, quick_v, coord)) for i in range(8)]
        train_thread = threading.Thread(target=train,
                args=(sess, train_op, coord, summary_op, writer, global_step))
        for thread in t:
            thread.start()
        train_thread.start()

        while True:
            try:
                time.sleep(10)
                saver.save(sess, os.path.join(args.logdir, "model"),
                        global_step=global_step.eval())
            except:
                coord.request_stop()
                coord.join([train_thread] + t)
                exit()


def board_visualizer():
    b = chess.Board()
    white, black = convert_to_array(b)
    plt.subplot(121)
    plt.imshow(white, cmap='Greys')
    plt.subplot(122)
    plt.imshow(black, cmap='Greys')
    plt.show()


if __name__ == '__main__':
    main()

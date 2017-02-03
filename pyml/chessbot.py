from __future__ import print_function
from utils import *
import chess
import argparse
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import os
import requests


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", default='tflogs/chess/linear_errorv2',
            type=str, help="Location to store tensorflow logs")

    parser.add_argument("--batch_size", default=32, type=int,
            help="Batch size to use when training")

    args = parser.parse_args()
    return args


def convert_to_array(board):
    P_T_I = {'k' : 0,
             'q' : 1,
             'p' : 2,
             'b' : 3,
             'r' : 4,
             'n' : 5}

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
            black[c][r] = P_T_I[p.lower()] + 6
            c += 1
            continue
        else:
            black[c][r] = P_T_I[p]
            white[c][r] = P_T_I[p] + 6
            c += 1
            continue

    return (white, np.rot90(black, k=2))


def play_self(sess, enqueue, X, Y, V, coord, verbose=False, log_dir=None):
    SAVE_TIME = 0
    b = chess.Board()
    b.reset()
    white_states = []
    black_states = []
    prev_winner = True
    s_t = time.clock()
    files = 0
    while not coord.should_stop():
        if b.is_game_over():
            r = b.result()
            if '1/2-1/2':
                print("Stalemate")
            else:
                if '1-0' == r:
                    ww = 1
                    bw = 0
                if '0-1' == r:
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
            if log_dir is not None and time.clock() - s_t > SAVE_TIME:
                from chess import pgn
                game = pgn.Game.from_board(b)
                print(game,
                        file=open(os.path.join(log_dir,
                            'test-{}.pgn'.format(files)), 'w'), end='\n\n')
                files += 1
                s_t = time.clock()
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
            b.push(curmove)
            white, black = convert_to_array(b)
            white_states.append(white)
        else:
            curmove = select_move(b.turn, None)
            b.push(curmove)
            white, black = convert_to_array(b)
            black_states.append(black)


def data_loading_thread(sess, enqueue, X, Y, coord):
    gen = load_data()
    while not coord.should_stop():
        w, b, r = next(gen)
        if r == '1-0':
            ww = 1
            bw = 0
        if r == '0-1':
            ww = 0
            bw = 1
        sess.run(enqueue, feed_dict={X: w, Y: [ww] * len(w)})
        sess.run(enqueue, feed_dict={X: b, Y: [bw] * len(b)})


def train(sess, train, coord, summaries, writer, global_step):
    while not coord.should_stop():
        _, s = sess.run([train, summaries])
        writer.add_summary(s, sess.run(global_step))
        writer.flush()


def build_net(X, histograms=False):
    net = add_conv_layer(X, [3, 3, 256], 'conv1')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_1', net)

    for i in range(8):
        with tf.variable_scope('resnet_{}'.format(i+2)):
            net = tf.contrib.layers.batch_norm(net, 0.99)
            I = net
            net = add_conv_layer(net, [3, 3, 256], 'conv_0')
            net = tf.nn.elu(net)
            net = add_conv_layer(net, [3, 3, 256], 'conv_1')

            net = tf.nn.elu(net) + I
            # if histograms:
                # tf.summary.histogram('conv_{}'.format(i+2), net)

    net = flatten(net)
    net = add_layer(net, 1, 'fc')
    if histograms:
        tf.summary.histogram('output_value', net)
    return net


def main():
    args = get_args()
    with tf.name_scope('DataManagement'):
        data_queue = tf.RandomShuffleQueue(60000, 1024, [tf.int64, tf.float32],
                [[8, 8], 1])
        X = tf.placeholder(tf.int64, [None, 8, 8], name='input')
        L_d = tf.placeholder(tf.float32, [None], name='win_status')
        enqueue_op = data_queue.enqueue_many([X, tf.expand_dims(L_d, axis=1)])
        B, L = data_queue.dequeue_many(args.batch_size)

    converted_board = tf.one_hot(B, 12)

    with tf.variable_scope("valuenet") as scope:
        value = build_net(converted_board, histograms=True)

    with tf.variable_scope("valuenet", reuse=True):
        quick_v = build_net(tf.one_hot(X, 12))

    with tf.name_scope('metrics'):
        entropy_error = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(value, L))

        abs_error = tf.reduce_mean(tf.abs(value - (L * 200 - 100)))
        error = abs_error
        tf.summary.scalar("Error", error)

        accuracy = tf.reduce_mean(tf.cast(tf.greater(tf.sign(value) * tf.sign(L - 0.5), 0),
                tf.float32))
        tf.summary.scalar("Accuracy", accuracy)

    global_step = tf.Variable(0, trainable=False)

    coord = tf.train.Coordinator()
    with tf.name_scope("gradients"):
        MAX_DELAY = 200
        delay = tf.Variable(MAX_DELAY, trainable=False)
        lr = tf.Variable(0.001, trainable=False)
        tf.summary.scalar("Learning_rate", lr)
        opt = tf.train.GradientDescentOptimizer(lr)
        gvs = opt.compute_gradients(error)
        tf.summary.scalar('gradient_norm', tf.global_norm(list(zip(*gvs))[0]))

        clipped_gvs = [(tf.clip_by_norm(g, 5.0), v) for g, v in gvs]
        grad_descent_op = opt.apply_gradients(clipped_gvs, global_step=global_step)

        ema = tf.train.ExponentialMovingAverage(0.99)
        maintain_averages = ema.apply([error])
        prev_error = tf.Variable(10.0)
        tf.summary.scalar("shadow_error", ema.average(error))

        new_lr, new_delay, new_error = lr, delay, prev_error
        # tf.cond(delay > 0,
                # lambda: [lr, delay-1, prev_error],
                # lambda: [
                    # tf.cond(ema.average(error) > 0.95 * prev_error,
                        # lambda: lr/2.0,
                        # lambda: lr),
                    # tf.Variable(MAX_DELAY, trainable=False),
                    # ema.average(error)])
        reduce_lr = tf.group(
                tf.assign(lr, new_lr),
                tf.assign(delay, new_delay),
                tf.assign(prev_error, new_error))

        with tf.control_dependencies([grad_descent_op]):
            train_op = tf.group(maintain_averages, reduce_lr)

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

        t = []
        # t = [threading.Thread(target=play_self,
            # args=(sess, enqueue_op, X, L_d, quick_v, coord)) for i in range(7)]
        t.append(threading.Thread(target=play_self, args=(sess, enqueue_op, X,
            L_d, quick_v, coord), kwargs={"log_dir": "chesslogs"}))
        t.append(threading.Thread(target=data_loading_thread, args=(sess,
            enqueue_op, X, L_d, coord)))
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


def _gather_data(out_dir=None):
    from lxml import html, etree
    HTML_LOCATION='/home/peachball/D/git/ML-Papers/datasets/chess/gambit.html'
    if out_dir is None:
        SAVE_LOC = '/home/peachball/D/git/ML-Papers/datasets/chess/'
    else:
        SAVE_LOC = out_dir
    HOST="http://www.gambitchess.com/"
    tree = html.parse(HTML_LOCATION)
    elems = tree.xpath('.//a')
    fem = list(filter(lambda x: x.text != 'pgn', elems))

    i = 0
    for f in fem:
        link = f.get('href')[3:]
        l = HOST + link
        filename = os.path.join(SAVE_LOC, "game{}.zip".format(i))
        i += 1
        r = requests.get(l, stream=True)
        print("\rDownloading {}/{}".format(i, len(fem)), end="")
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
    print()


def _unzip_dir(data_dir=None):
    import zipfile
    if data_dir is None:
        data_dir = '/home/peachball/D/git/ML-Papers/datasets/chess'
    files = os.listdir(data_dir)

    for f in files:
        filename = os.path.join(data_dir, f)
        try:
            with zipfile.ZipFile(filename, 'r') as zipref:
                zipref.extractall(data_dir)
        except:
            print("Unable to unzip file: {}".format(filename))


def load_data(data_dir=None, stalemates=False):
    from chess import pgn
    if data_dir is None:
        data_dir = '/home/peachball/D/git/ML-Papers/datasets/chess/pgn'

    files = os.listdir(data_dir)

    def read_game(handler):
        try:
            game = pgn.read_game(handler)
        except:
            print("Unable to read game")
            game = None
        finally:
            return game

    while True:
        for f in files:
            filename = os.path.join(data_dir, f)
            handler = open(filename)
            game = read_game(handler)

            while game is not None:
                result = game.headers['Result']
                if result == '1/2-1/2':
                    game = read_game(handler)
                    continue
                white_states = []
                black_states = []

                while not game.is_end():
                    game = game.variations[0]
                    b = game.board()

                    whi, blk = convert_to_array(b)
                    if b.turn:
                        black_states.append(blk)
                    else:
                        white_states.append(whi)

                if len(white_states) == 0 or len(black_states) == 0:
                    game = read_game(handler)
                    continue

                yield (white_states, black_states, result)
                del white_states[:]
                del black_states[:]
                game = read_game(handler)


if __name__ == '__main__':
    main()

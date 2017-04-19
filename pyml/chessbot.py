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
import random
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", default='tflogs/chess/temporaldiff',
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
    white = np.zeros((8, 8)) + 12
    black = np.zeros((8, 8)) + 12

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

    return (white, np.fliplr(black))


def play_self(sess, enqueue, X, Y, V, coord, W, verbose=False, log_dir=None,
        engine=None, sw=None, thread_id=0):
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
            if r == '1/2-1/2':
                ww = 0
                bw = 0
            if '1-0' == r:
                ww = 100
                bw = -100
            if '0-1' == r:
                ww = -100
                bw = 100
            print("Played game of length {}, {}"
                    .format(len(white_states), r))
            w_vals = np.concatenate((
                    sess.run(V, feed_dict={X: np.array(white_states)})[1:, 0],
                    np.array(ww)[None]))
            b_vals = np.concatenate((
                    sess.run(V, feed_dict={X: np.array(black_states)})[1:, 0],
                    np.array(bw)[None]))
            if sw is not None:
                game_summary = tf.Summary(
                        value=[tf.Summary.Value(tag="Game length",
                            simple_value=len(white_states))])
                sw.add_summary(game_summary)
            sess.run(enqueue, feed_dict={X: np.array(white_states), Y: w_vals,
                W: [ww] * len(white_states)})
            sess.run(enqueue, feed_dict={X: np.array(black_states), Y: b_vals,
                W: [bw] * len(black_states)})

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

        def minmax(board, depth, white=True, getmove=False, top=5):
            # param white is "is it whites turn?"
            if depth == 0:
                assert(not getmove)
                wh, bl = convert_to_array(board)
                if white:
                    return -sess.run(V, feed_dict={X: bl[None]})
                else:
                    return -sess.run(V, feed_dict={X: wh[None]})

            moves = list(board.legal_moves)

            states = []
            for m in moves:
                board.push(m)
                wh, bl = convert_to_array(board)
                if white:
                    states += [wh]
                else:
                    states += [bl]
                board.pop()
            values = sess.run(V, feed_dict={X: np.array(states)})
            top_n = np.argsort(values, axis=0)[-top:]
            subscore = -1e8
            best_move = None
            for ind in top_n:
                i = int(ind)
                board.push(moves[int(ind)])
                score = -minmax(board, depth - 1, white=not white)
                board.pop()
                if score > subscore:
                    subscore = score
                    best_move = moves[int(ind)]

            if getmove:
                return best_move
            else:
                return subscore

        def select_move(white=True, T=-1, use_engine=False):
            if T == 'minmax':
                return minmax(b, 1, white=white, getmove=True, top=2)
            if use_engine and (not engine is None):
                engine.position(b)
                return engine.go()[0]

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
            if thread_id % 2 == 0:
                curmove = select_move(b.turn, use_engine=True)
            else:
                curmove = select_move(b.turn, "minmax")
            b.push(curmove)
            white, black = convert_to_array(b)
            white_states.append(white)
        else:
            if thread_id % 2 == 1:
                curmove = select_move(b.turn, use_engine=True)
            else:
                curmove = select_move(b.turn, "minmax")
            b.push(curmove)
            white, black = convert_to_array(b)
            black_states.append(black)


def data_loading_thread(sess, enqueue, X, Y, coord, belikepro=True, delay=5):
    gen = load_data()
    while not coord.should_stop():
        w, b, r = next(gen)
        assert r != '1/2-1/2'

        if r == '1-0':
            ww = 1
            bw = 0
        if r == '0-1':
            ww = 0
            bw = 1
        if belikepro:
            ww = 1
            bw = 1

        sess.run(enqueue, feed_dict={X: w, Y: [ww] * len(w)})
        sess.run(enqueue, feed_dict={X: b, Y: [bw] * len(b)})
        time.sleep(5)


def train(sess, train, coord, summaries, writer, global_step):
    while not coord.should_stop():
        _, s = sess.run([train, summaries])
        writer.add_summary(s, sess.run(global_step))
        writer.flush()


def build_net(X, histograms=False):
    net = add_conv_layer(X, [3, 3], 64, 'conv1')
    net = tf.nn.elu(net)
    if histograms:
        tf.summary.histogram('conv_1', net)

    for i in range(15):
        with tf.variable_scope("resnet{}".format(i)):
            I = net
            net = add_conv_layer(net, [3, 3], 64, 'conv{}_0'.format(i+2))
            net = tf.contrib.layers.batch_norm(net, 0.99)
            net = tf.nn.elu(net)
            if histograms:
                tf.summary.histogram("conv{}_0".format(i + 2), net)
            net = add_conv_layer(net, [3, 3], 64, 'conv{}_1'.format(i+2))
            net = tf.contrib.layers.batch_norm(net, 0.99)

            net = tf.nn.elu(net) + I
            if histograms:
                tf.summary.histogram("conv{}_1".format(i + 2), net)

    net = flatten(net)
    features = net
    net = add_layer(features, 1, 'fc')
    if histograms:
        tf.summary.histogram('output_value', net)
    return net


def get_model_params(X, L, cpu=0):
    device = '/cpu:{}'.format(cpu)
    with tf.device(device):
        with tf.variable_scope('valuenet', reuse=True):
            return build_net(tf.one_hot(X, 13))


def main():
    args = get_args()
    with tf.name_scope('DataManagement'):
        data_queue = tf.RandomShuffleQueue(60000, 128, [tf.int64, tf.float32, tf.int32],
                [[8, 8], 1, 1])
        X = tf.placeholder(tf.int64, [None, 8, 8], name='input')
        L_d = tf.placeholder(tf.float32, [None], name='labeled_value')
        W = tf.placeholder(tf.int32, [None], name='win_status')
        enqueue_op = data_queue.enqueue_many([X, tf.expand_dims(L_d, axis=1),
            tf.expand_dims(W, axis=1)])
        B, L, b_W = data_queue.dequeue_many(args.batch_size)

    converted_board = tf.one_hot(B, 13)

    with tf.device('/gpu:0'):
        with tf.variable_scope("valuenet") as scope:
            value = build_net(tf.stop_gradient(converted_board), histograms=True)

    q_vs = [get_model_params(X, L_d, cpu=i) for i in range(8)]

    with tf.name_scope('metrics'):
        entropy_error = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=value, labels=L))

        abs_error = tf.reduce_mean(tf.abs(value - (L * 200 - 100)))
        mse_error = tf.reduce_mean(tf.squared_difference(value, L))
        regularizer = tf.add_n([tf.nn.l2_loss(v) for v in
            tf.trainable_variables() if not "bias" in v.name]) * 0.001
        error = mse_error
        tf.summary.scalar("Error", error)
        tf.summary.histogram("Target Values", L)

        accuracy = tf.reduce_mean(tf.cast(
                tf.greater(
                    tf.sign(value) * tf.sign(tf.cast(b_W, tf.float32)), 0),
                tf.float32))
        tf.summary.scalar("Accuracy", accuracy)

    global_step = tf.Variable(0, trainable=False)

    coord = tf.train.Coordinator()
    with tf.device('/gpu:0'):
        with tf.name_scope("gradients"):
            MAX_DELAY = 200
            delay = tf.Variable(MAX_DELAY, trainable=False)
            lr = tf.Variable(0.003, trainable=False)
            tf.summary.scalar("Learning_rate", lr)
            opt = tf.train.AdamOptimizer(1e-4)
            gvs = opt.compute_gradients(error)
            tf.summary.scalar('gradient_norm', tf.global_norm(list(zip(*gvs))[0]))

            clipped_gvs = [(tf.clip_by_norm(g, 5.0), v) for g, v in gvs]
            grad_descent_op = opt.apply_gradients(clipped_gvs, global_step=global_step)

            ema = tf.train.ExponentialMovingAverage(0.99)
            maintain_averages = ema.apply([error])
            prev_error = tf.Variable(10.0)
            tf.summary.scalar("shadow_error", ema.average(error))

            # lr, delay, prev_error
            new_lr, new_delay, new_error = tf.cond(delay > 0,
                    lambda: [lr, delay-1, prev_error],
                    lambda: [
                        tf.cond(ema.average(error) > 0.95 * prev_error,
                            lambda: lr/2.0,
                            lambda: lr),
                        tf.Variable(MAX_DELAY, trainable=False),
                        ema.average(error)])
            reduce_lr = tf.group(
                    tf.assign(lr, new_lr),
                    tf.assign(delay, new_delay),
                    tf.assign(prev_error, new_error))

            with tf.control_dependencies([grad_descent_op]):
                train_op = tf.group(maintain_averages, reduce_lr)

    config = tf.ConfigProto(device_count={'CPU': 8, 'GPU': 1},
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=1,
            allow_soft_placement=True)
    init_op = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.logdir)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        ckpt = tf.train.latest_checkpoint(args.logdir)
        try:
            saver.restore(sess, ckpt)
        except:
            sess.run(init_op)
            print("Unable to load previous checkpoint")
            writer.add_graph(sess.graph, global_step=global_step.eval())

        t = []
        t = [threading.Thread(target=play_self,
            args=(sess, enqueue_op, X, L_d, v, coord, W),
            kwargs={'engine': load_engine(), 'thread_id': i, 'sw': writer})
            for i, v in enumerate(q_vs[:-1])]
        t.append(threading.Thread(target=play_self, args=(sess, enqueue_op, X,
            L_d, q_vs[-1], coord, W), kwargs={"log_dir": "chesslogs", "engine":
                load_engine(), 'sw': writer}))
        # t.append(threading.Thread(target=data_loading_thread, args=(sess,
            # enqueue_op, X, L_d, coord)))
        train_thread = threading.Thread(target=train,
                args=(sess, train_op, coord, summary_op, writer, global_step))

        for thread in t:
            thread.start()
        train_thread.start()

        while True:
            try:
                time.sleep(10)
                print("Data queue size:", sess.run(data_queue.size()))
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


def load_data(data_dir=None):
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


def load_engine(location=None):
    from chess import uci
    if location is None:
        if os.name == 'posix':
            location = './stockfish_8_x64'
        if os.name == 'nt':
            location = 'stockfish_8_x64.exe'
    assert(location is not None)
    engine = uci.popen_engine(location)
    engine.uci()

    return engine


def naiveminmax(board, depth, getbestmove=False):
    assert(depth <= 5)
    if depth == 0:
        assert(not getbestmove)
        if len(board.legal_moves) != 0:
            return random.uniform(-10, 10)
        return -10

    best_score = -10
    best_move = None
    for mv in board.legal_moves:
        board.push(mv)
        new_score = -naiveminmax(board, depth - 1)
        board.pop()
        if new_score > best_score:
            best_score = new_score
            best_move = mv

    if getbestmove:
        return best_move
    else:
        return best_score


def test_naive_minmax(filename="chesslogs/naiveminmax.pgn"):
    from chess import pgn
    board = chess.Board()
    mv = naiveminmax(board, 2, True)
    while mv is not None and not board.is_game_over():
        board.push(mv)
        mv = naiveminmax(board, 2, True)
        if mv is None:
            break
        print("Moved: ", mv.uci())
    print("Game status:", board.result())
    f = open(filename, 'w')
    print(pgn.Game.from_board(board), file=f)


def run_uci_mode():
    if input() != 'uci':
        exit()
    board = chess.Board()
    while True:
        cmd = input()
        args = cmd.split(' ')
        if args[0] == 'isready':
            print('readyok')
        if args[0] == 'setoption':
            pass
        if args[0] == 'ucinewgame':
            pass
    pass


if __name__ == '__main__':
    main()

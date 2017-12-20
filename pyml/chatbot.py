#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
from datautils import *


class ChatBot():
    def __init__(self, vocab_size=1024, word_size=16):
        self.inp = tf.placeholder(tf.int32, [None, None])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.inp_length = tf.placeholder(tf.int32, [None])
        self.lab_length = tf.placeholder(tf.int32, [None])

    def _build_chatbot(self, inp, vocab_size, word_size, sentence_lengths=None):
        V = tf.get_variable("vocabulary", [vocab_size, word_size], tf.float32,
                initializer=tf.random_normal_initializer())
        embed = tf.nn.embedding_lookup(V, inp)
        lstm = tf.contrib.rnn.LSTMCell(256)
        self.lstm_init_state = lstm.zero_state(tf.shape(inp)[0])
        outputs, l_states = tf.nn.dynamic_rnn(lstm, inp, sequence_length=sentence_lengths,
            initial_state=self.lstm_init_state)


class ConvSeq2Seq():
    def __init__(self):
        pass


class Seq2Seq():
    def __init__(self, X_s, Y_s, vocab_size, X_len=None, Y_len=None,
            embedding_size=256):
        """
            params:
            X_s: [batch_size, max_time_steps, embedding_size]
            X_len: [batch_size,]
        """
        self.input = X_s
        self.V = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
            -1.0, 1.0), name="vocab")
        self.vocab_size = vocab_size
        self.goal = Y_s
        if X_len is None:
            s = tf.shape(X_s)
            self.inp_len = tf.fill([s[0]], s[1])
        else:
            self.inp_len = X_len
        if Y_len is None:
            s = tf.shape(X_s)
            self.out_len = tf.fill([s[0]], s[1])
        else:
            self.out_len = Y_len
        self.embedded_input = tf.nn.embedding_lookup(self.V, X_s)
        self.embedded_goal = tf.nn.embedding_lookup(self.V, self.goal)
        self.build_net(self.embedded_input)


    def build_net(self, inp):
        """
            params:
            inp: [batch_size, max_time_steps, embedding_size]
        """
        with tf.variable_scope("encoder"):
            lstm = self._get_cell()
            self.lstm = lstm
            bs = tf.shape(inp)[0]
            self.enc_init_state = lstm.zero_state(bs, tf.float32)
            output, self.end_enc_state = tf.nn.dynamic_rnn(lstm, inp,
                    sequence_length=self.inp_len, initial_state=self.enc_init_state)

        self.encoded_vals = output
        with tf.variable_scope("decoder"):
            dec = self._get_cell()
            self.dec = dec
            output, self.end_dec_state = tf.nn.dynamic_rnn(dec, self.embedded_goal,
                    initial_state=self.end_enc_state)
            self.decoded_logits = tf.layers.dense(output, self.vocab_size,
                    name="projection")
        self.predict_word()


    def _get_cell(self):
        return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(256) for i
            in range(4)])


    def generate_sentence(self, maxlen):
        batch_size = tf.shape(self.embedded_input)[0]
        s = tf.zeros([batch_size], dtype=tf.int32)
        s = tf.nn.embedding_lookup(self.V, s)
        lstm_state = self.end_enc_state
        seq = [s]
        words = []
        for i in range(maxlen):
            new_state, lstm_state = self.dec(seq[-1], lstm_state)
            word_id = tf.argmax(new_state, axis=1)
            new_word = tf.nn.embedding_lookup(self.V, word_id)
            seq.append(new_word)
            words.append(word_id)
        return tf.concat([tf.expand_dims(s, axis=1) for s in words], axis=1)


    def predict_word(self):
        lstm_state = self.end_enc_state
        inp = tf.squeeze(self.embedded_input, axis=1)
        self.n_word, self.n_word_state = self.dec(inp, self.end_enc_state)


def train_chatbot(LOGDIR="tflogs/smallseq2seq",
        MODEL_FILE="../datasets/cornell movie-dialogs corpus/8k.model"):
    x = tf.placeholder(tf.int32, [None, None], name="input_sentences")
    sentence_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
    o = tf.placeholder(tf.int32, [None, None], name="gal_sentences")
    out_lengths = tf.placeholder(tf.int32, [None], name='out_len')
    with tf.variable_scope("model") as scope:
        m = Seq2Seq(x, o, 8000, sentence_lengths, out_lengths)
        scope.reuse_variables()
    convs = load_cornell_corpus(0, 1e10)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=o[:,1:],
            logits=m.decoded_logits[:,:-1,:])
    output_sentence = tf.argmax(m.decoded_logits, axis=-1)
    error = tf.reduce_sum(cross_entropy)
    tf.summary.scalar("error", error)
    global_step = tf.Variable(0, name="global_step")
    train_op = tf.train.RMSPropOptimizer(1e-3).minimize(error,
            global_step=global_step)

    summary_op = tf.summary.merge_all()
    sw = tf.summary.FileWriter(LOGDIR)
    sv = tf.train.Supervisor(logdir=LOGDIR,
            summary_op=None,
            summary_writer=None,
            save_model_secs=30)
    with sv.managed_session() as sess:
        sw.add_graph(sess.graph)
        d = get_batch(convs)
        i = -1
        while True:
            data = next(d)
            sent, go, slen, glen = data[0], data[1], data[2], data[3]
            s, e, _ = sess.run([summary_op, error, train_op],
                    feed_dict={x: sent,
                        sentence_lengths: slen,
                        o: go,
                        out_lengths: glen})
            sw.add_summary(s, global_step=sess.run(global_step))
            i += 1
            if i % 100 == 0:
                config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
                embedding = config.embeddings.add()
                embedding.tensor_name = m.V.name
                embedding.metadata_path = "/home/peachball/D/git/ML-Papers/pyml/8k.vocab"
                tf.contrib.tensorboard.plugins.projector.visualize_embeddings(sw, config)
                inp_s = sent[:1,:]
                inp_l = slen[:1]
                print("Input sentence:")
                s = decode_spm(' '.join(map(lambda x:str(x), inp_s[0,:inp_l[0]])),
                        MODEL_FILE)
                print(s)

                gen_s = [np.array([[0]])]
                s = sess.run(m.end_enc_state, feed_dict={x: inp_s,
                    sentence_lengths: inp_l})
                while gen_s[-1][0,-1] != 2 and len(gen_s) < 100:
                    s, d_log = sess.run([m.end_dec_state, m.decoded_logits],
                            feed_dict={m.end_enc_state: s, o: gen_s[-1]})
                    word_id = np.argmax(d_log, axis=-1)
                    gen_s.append(word_id)
                gen_s = np.array(gen_s)
                gen_s = gen_s.squeeze()

                print("Bot response:")
                print(decode_spm(' '.join(map(lambda x: str(x), gen_s)),
                    MODEL_FILE))
                print(gen_s)
                # print("Actual response:")
                # print(decode_spm(' '.join(map(lambda x: str(x), go[0,:glen[0]])),
                    # MODEL_FILE))
                # print(go[0,:glen[0]])
                # print(gen_s[0,1:])


def interactive_chatbot(LOGDIR="tflogs/seq2seqchatbot",
        MODEL_FILE="../datasets/cornell movie-dialogs corpus/8k.model"):
    x = tf.placeholder(tf.int32, [None, None], name="input_sentences")
    sentence_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
    o = tf.placeholder(tf.int32, [None, None], name="gal_sentences")
    out_lengths = tf.placeholder(tf.int32, [None], name='out_len')
    with tf.variable_scope("model") as scope:
        m = Seq2Seq(x, o, 8000, sentence_lengths, out_lengths)
    output_sentence = tf.argmax(m.decoded_logits, axis=-1)
    global_step = tf.Variable(0, name="global_step")

    sv = tf.train.Supervisor(logdir=LOGDIR,
            summary_op=None,
            summary_writer=None,
            save_model_secs=30)
    with sv.managed_session() as sess:
        while True:
            inp_s = encode_spm(input().strip(), MODEL_FILE)
            print(inp_s)
            print("Input sentence:")
            s = decode_spm(' '.join(list(map(lambda x: str(x), inp_s))), MODEL_FILE)
            print(s)
            inp_s = [inp_s]

            gen_s = [np.array([[0]])]
            s = sess.run(m.end_enc_state, feed_dict={x: inp_s, sentence_lengths:
                [len(inp_s)]})
            while gen_s[-1][0,-1] != 2 and len(gen_s) < 100:
                s, d_log = sess.run([m.end_dec_state, m.decoded_logits],
                        feed_dict={m.end_enc_state: s, o: gen_s[-1]})
                word_id = np.argmax(d_log, axis=-1)
                gen_s.append(word_id)
            gen_s = np.array(gen_s)
            gen_s = gen_s.squeeze()

            print("Bot response:")
            print(decode_spm(' '.join(map(lambda x: str(x), gen_s)),
                MODEL_FILE))
            print(gen_s)


def get_batch(convs, batch_size=32):
    o = [[], [], [], []]
    while True:
        for i in range(len(convs)):
            for j in range(len(convs[i]) - 1):
                o.append((convs[i][j], convs[i][j+1]))
                o[0].append(convs[i][j])
                o[1].append([0] + convs[i][j+1])
                o[2].append(len(convs[i][j]))
                o[3].append(len(convs[i][j+1])+1)
                if len(o) >= batch_size:
                    # Normalize lengths
                    # print(o[0])
                    for k in range(2):
                        maxlen = max(len(p) for p in o[k])
                        o[k] = [pl + [0] * (maxlen - len(pl)) for pl in o[k]]

                    o[0] = np.array(o[0])
                    o[1] = np.array(o[1])
                    o[2] = np.array(o[2])
                    o[3] = np.array(o[3])
                    yield o
                    o = [[], [], [], []]


class SequenceClassifier():
    def __init__(self, X, x_len=None, vocab_size=8000, embedding_size=16):
        self.input = X
        self.vocab_size = vocab_size
        if x_len is None:
            s = tf.shape(self.input)
            self.sent_length = tf.fill([s[0]], s[1])
        else:
            self.sent_length = x_len
        self.V = tf.Variable(tf.random_normal([vocab_size, embedding_size]))
        self.embedded_input = tf.nn.embedding_lookup(self.V, self.input)


    def build_net(self, inp):
        lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(256) for x
            in range(3)])
        self.init_state = lstm.zero_state(tf.shape(inp)[0], tf.float32)
        outputs, self.end_state = tf.nn.dynamic_rnn(lstm, self.embedded_input,
                sequence_length=self.sent_length, initial_state=self.init_state)
        net_out = tf.layers.dense(outputs[:,-1,:], self.vocab_size)
        self.output = net_out


def sentiment_analysis():
    (x_p, x_n) = load_imdb_sentiment()
    def get_sent_len(v):
        return list(map(lambda l: len(l), v))
    x_p_len = get_sent_len(x_p)
    x_n_len = get_sent_len(x_n)


if __name__ == '__main__':
    train_chatbot()

import numpy as np
import keras
import ast
import codecs
import os
import subprocess
import itertools
import PIL


def _load_hehexd_image(index, dataset='hehexdDataSet', image_dim=(512, 512)):
    from os.path import join, isfile
    from os import listdir
    dataset = join('..', 'datasets', dataset)
    files = [f for f in listdir(dataset) if isfile(join(dataset, f))]
    if index >= len(files):
        index = index % len(files)

    from PIL import Image
    img = Image.open(join(dataset, files[index]))
    img = img.convert('RGB')
    img = img.resize(image_dim)
    return np.array(img)


def load_hehexd_dataset(range_tuple, dataset='hehexdDataSet'):
    imgs = []
    for i in range(range_tuple[0], range_tuple[1]):
        imgs.append(_load_hehexd_image(i, dataset=dataset))

    return np.array(imgs).astype('float32') / 255.0


def load_mnist():
    from keras.datasets import mnist
    (x, y), _ = mnist.load_data()
    return (x, y)


def load_cifar10():
    from keras.datasets import cifar10
    (x, y), _ = cifar10.load_data()
    return (x, y)


def load_cifar100():
    from keras.datsets import cifar100
    (x, y), _ = cifar100.load_data()
    return (x, y)


def load_imdb_sentiment(data_dir="../datasets/aclImdb/train/8k/",
        pos="8kpos.txt", neg="8kneg.txt"):
    pf = os.path.join(data_dir, pos)
    nf = os.path.join(data_dir, neg)
    def readfile(f):
        d = []
        for l in f:
            d.append([int(x) for x in l.strip().split(' ')])
        return d
    pdata = readfile(open(pf, 'r'))
    ndata = readfile(open(nf, 'r'))
    return (pdata, ndata)


def load_cornell_corpus(start=0, end=0):
    CORNELL_DATASET_LOCATION='../datasets/cornell movie-dialogs corpus'

    LINES = None

    def _process_file(parsed_location='8klines.txt'):
        nonlocal LINES
        if LINES is not None:
            return LINES
        LINES = {}
        with codecs.open(os.path.join(
            CORNELL_DATASET_LOCATION, "movie_lines.txt"), 'r', 'iso-8859-1') as f:
            pf = open(os.path.join(CORNELL_DATASET_LOCATION, parsed_location),
                    'r')
            for l, pl in zip(f, pf):
                d = l.split(' +++$+++ ')
                LINES[d[0]] = [int(x) for x in pl.strip().split(' ')]
            # pf.seek(0)
            # f.seek(0)
            # ptf.shape(inp)[0],rint(sum(1 for l in f), "vs", sum(1 for l in pf))
        return LINES


    def read_conversations(start=0, end=0):
        assert end > start
        with open(os.path.join(
            CORNELL_DATASET_LOCATION,
            'movie_conversations.txt'), 'r') as f:
            convs = []
            ld = _process_file()
            for i, l in enumerate(f):
                if i > end and end > 0:
                    break
                convs += [ast.literal_eval(l.split(' +++$+++ ')[-1])]
            o = []
            for c in convs:
                try:
                    o.append([ld[l] for l in c])
                except KeyError:
                    pass
            return o
    return read_conversations(start=start, end=end)


def train_spm(input_data, model_prefix, vocab_size):
    subprocess.call(['spm_train',
        '--input={}'.format(input_data),
        '--model_prefix={}'.format(model_prefix),
        '--vocab_size={}'.format(vocab_size)])


def encode_spm(text, model_file):
    p = subprocess.Popen(["spm_encode", "--model={}".format(model_file),
        "--output_format=id"], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
        stderr=subprocess.PIPE)
    encoded = p.communicate(input=bytes(text, 'utf-8'))[0]
    if len(encoded) > 0:
        return [int(x) for x in encoded.decode('utf-8').strip().split(' ')]
    return []


def decode_spm(arrs, model_file):
    p = subprocess.Popen(["spm_decode", "--model={}".format(model_file),
        "--input_format=id"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    decoded = p.communicate(input=bytes(arrs, 'utf-8'))[0]
    return decoded.decode('utf-8')


def ponydata_generator(batch_size,
        data_dir="../datasets/equestriadaily/distorted_64x64"):
    from PIL import Image
    raise NotImplementedError("Lol I'm lazy")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = load_hehexd_dataset((0, 10))
    plt.imshow(x[0])
    plt.show()

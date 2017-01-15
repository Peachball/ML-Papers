from PIL import Image
import requests
import csv
import os
import collections
import numpy as np
import io

DATA_DIR='/home/peachball/D/git/EE-DeepLearning/datasets/openimages'

def get_image(url):
    response = requests.get(url)
    stream = io.BytesIO(response.content)
    img = Image.open(stream)
    return img


def _load_dictionary(data_dir):
    dict_dir = os.path.join(data_dir, 'dict.csv')
    reader = csv.reader(open(dict_dir), delimiter=',', quotechar='"')

    id_to_word = collections.OrderedDict()
    index = 0
    for r in reader:
        id_to_word[r[0]] = (r[1], index)
        index += 1

    print(len(id_to_word))
    return id_to_word


def _move_readers(i, l, n):
    if n <= 0:
        return
    index = 0
    p = None
    for r in i:
        img_id = r[0]
        for b in l:
            if b[0] != img_id:
                p = b
                break
        index += 1
        if index >= n:
            return p


def next_images_batch(batch_size, data_dir=DATA_DIR, train=True, local=False,
        size=(224, 224), seek=0, stride=1):
    if train:
        label_dir = os.path.join(data_dir, 'machine_ann', 'train', 'labels.csv')
        image_dir = os.path.join(data_dir, 'images', 'train', 'images.csv')
    else:
        label_dir = os.path.join(data_dir, 'machine_ann', 'validation',
        'labels.csv')
        image_dir = os.path.join(data_dir, 'images', 'validation', 'images.csv')

    img_f = open(image_dir)
    label_f = open(label_dir)
    img_url_reader = csv.reader(img_f, delimiter=',', quotechar='"')
    label_reader = csv.reader(label_f, delimiter=',', quotechar='"')
    next(img_url_reader)
    next(label_reader)

    dictionary = _load_dictionary(data_dir)
    def indexof(l):
        return dictionary[l][1]

    prev_line = _move_readers(img_url_reader, label_reader, seek)

    image_array = np.zeros((batch_size,) + size + (3,))
    label_array = np.zeros((batch_size, len(dictionary)))
    i = 0
    for r in img_url_reader:
        img_id = r[0]
        img_url = r[2]
        img = get_image(img_url)
        img = img.resize(size)
        image_array[i] = np.array(img)

        labels = []
        if prev_line is not None:
            if prev_line[0] == img_id:
                label_array[i,indexof(prev_line[2])] = float(prev_line[3])

                for l in label_reader:
                    if l[0] != img_id:
                        prev_line = l
                        break
                    label_array[i,indexof(l[2])] = float(l[3])

        if prev_line is None:
            for l in label_reader:
                if l[0] != img_id:
                    prev_line = l
                    break
                label_array[i,indexof(l[2])] = float(l[3])

        i += 1

        if i >= batch_size:
            yield (image_array.astype('float32') / 256.0,
                    label_array / label_array.sum(1)[:,None])
            prev_line = _move_readers(img_url_reader, label_reader, stride)
            i = 0


if __name__ == '__main__':
    g = next_images_batch(3, stride=5)
    gp = next_images_batch(3, seek=8)
    next(g)
    print(np.max(next(g)[0] - next(gp)[0]))

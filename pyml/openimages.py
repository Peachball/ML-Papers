from PIL import Image
import requests
import csv
import os
import collections
import numpy as np
import io

DATA_DIR='/home/peachball/D/git/EE-DeepLearning/datasets/openimages'

def get_image(url, savepath=None):
    response = requests.get(url, allow_redirects=False)
    stream = io.BytesIO(response.content)
    img = Image.open(stream)
    if savepath is not None:
        img.save(savepath)
    return img


def _load_dictionary(data_dir):
    dict_dir = os.path.join(data_dir, 'dict.csv')
    reader = csv.reader(open(dict_dir), delimiter=',', quotechar='"')

    id_to_word = {}
    index_to_word = {}
    index = 0
    for r in reader:
        id_to_word[r[0]] = (r[1], index)
        index_to_word[index] = (r[1], r[0])
        index += 1

    return (id_to_word, index_to_word)
dictionary, reverse_map = _load_dictionary(DATA_DIR)


def _move_readers(i, l, n):
    if n <= 0:
        return None
    index = 0
    p = None
    for r in i:
        img_id = r[0]
        exists = False

        if p is not None:
            if p[0] == img_id:
                exists = True

        for b in l:
            if b[0] != img_id:
                p = b
                break
            exists = True

        if exists:
            index += 1
        if index >= n:
            return p


def get_id(index):
    return reverse_map[index][1]


def get_name(index):
    return reverse_map[index][0]


def next_images_batch(batch_size, data_dir=DATA_DIR, train=True, local=False,
        size=(224, 224), seek=0, stride=1, savedir=None):
    global dictionary
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

    def indexof(l):
        return dictionary[l][1]

    prev_line = _move_readers(img_url_reader, label_reader, seek)

    image_array = np.zeros((batch_size,) + size + (3,))
    label_array = np.zeros((batch_size, len(dictionary)))
    i = 0
    index = 0
    for r in img_url_reader:
        img_id = r[0]

        labels = []
        exists = False
        if prev_line is not None:
            if prev_line[0] == img_id:
                exists = True
                label_array[i,indexof(prev_line[2])] = float(prev_line[3])

                for l in label_reader:
                    if l[0] != img_id:
                        prev_line = l
                        break
                    label_array[i,indexof(l[2])] = float(l[3])

        elif prev_line is None:
            for l in label_reader:
                if l[0] != img_id:
                    prev_line = l
                    break
                exists = True
                label_array[i,indexof(l[2])] = float(l[3])

        if exists:
            img_filename = None
            if savedir is not None:
                img_filename = os.path.join(savedir, img_id.replace('/', '-'))

            try:
                try:
                    img_url = r[10]
                    img = get_image(img_url, savepath=img_filename)
                except:
                    img_url = r[2]
                    img = get_image(img_url, savepath=img_filename)
                img = img.resize(size).convert('RGB')
                image_array[i] = np.array(img)
                i += 1
            except Exception as e:
                print("Unable to get image")
                print("Exception: ", e)
            index += 1

        if index >= batch_size:
            if stride > 0:
                prev_line = _move_readers(img_url_reader, label_reader, stride)
            index = 0

        if i >= batch_size:
            yield (image_array.astype('float32') / 256.0, label_array)
            image_array = np.zeros((batch_size,) + size + (3,))
            label_array = np.zeros((batch_size, len(dictionary)))
            i = 0


if __name__ == '__main__':
    g = next_images_batch(2, seek=1, stride=1)
    for i in range(2):
        _, l = next(g)
        print(get_id(int(l.argmax(axis=1)[1])))

import numpy as np
import keras

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = load_hehexd_dataset((0, 10))
    plt.imshow(x[0])
    plt.show()

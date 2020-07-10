import os
import pickle
from typing import Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

import common


def load_train(path: str, channel_first: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    x_batches = []
    y_batches = []
    for i in range(5):
        with open(os.path.join(path, 'data_batch_%d' % (i + 1)), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            x_batches.append(data_dict[b'data'])
            y_batches.append(data_dict[b'labels'])
    x = np.concatenate(x_batches, axis=0).reshape((50000, 3, 32, 32)).astype('float32')
    if not channel_first:
        x = np.transpose(x, axes=(0, 2, 3, 1))
    y = np.concatenate(y_batches, axis=0)
    return x, y


def load_test(path: str, channel_first: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(path, 'test_batch'), 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        x = data_dict[b'data'].reshape((10000, 3, 32, 32)).astype('float32')
        if not channel_first:
            x = np.transpose(x, axes=(0, 2, 3, 1))
        y = np.array(data_dict[b'labels'])
        return x, y


def get_train_iterator(x: np.ndarray, y: np.ndarray) -> image.NumpyArrayIterator:
    y = keras.utils.to_categorical(y, 10)
    return image.ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True,
        width_shift_range=4, height_shift_range=4, horizontal_flip=True,
        validation_split=0.1
    ).flow(x, y, batch_size=common.batch_size, shuffle=True)


def get_test_iterator(x: np.ndarray, y: np.ndarray) -> image.NumpyArrayIterator:
    y = keras.utils.to_categorical(y, 10)
    return image.ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True
    ).flow(x, y, batch_size=common.batch_size)


if __name__ == '__main__':
    x_train, y_train = load_train('cifar10', channel_first=False)
    train_iter = get_train_iterator(x_train, y_train)

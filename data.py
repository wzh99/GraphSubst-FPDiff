import os
import pickle
from typing import Tuple

import numpy as np
from tensorflow.keras.preprocessing import image

import common

val_split = 0.1


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


def get_train_val_iterator(x: np.ndarray, y: np.ndarray) -> \
        Tuple[image.NumpyArrayIterator, image.NumpyArrayIterator]:
    data_gen = image.ImageDataGenerator(
        samplewise_center=True, samplewise_std_normalization=True,
        width_shift_range=4, height_shift_range=4, horizontal_flip=True,
        validation_split=val_split
    )
    return (data_gen.flow(x, y, batch_size=common.batch_size, shuffle=True,
                          subset='training'),
            data_gen.flow(x, y, batch_size=common.batch_size, shuffle=False,
                          subset='validation'))


def get_test_iterator(x: np.ndarray, y: np.ndarray) -> image.NumpyArrayIterator:
    return image.ImageDataGenerator(
        samplewise_center=True, samplewise_std_normalization=True
    ).flow(x, y, batch_size=common.batch_size, shuffle=False)


class TvmDataGen:
    def __init__(self, x: np.ndarray, y: np.ndarray, dtype: str = common.dtype):
        self.dtype = dtype
        self.num_batches = len(x) // common.batch_size
        self.iter = image.ImageDataGenerator(
            samplewise_center=True, samplewise_std_normalization=True,
            data_format='channels_first'
        ).flow(x, y, batch_size=common.batch_size, shuffle=False)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.iter[idx]
        return x.astype(self.dtype), y


if __name__ == '__main__':
    x_train, y_train = load_train('cifar10', channel_first=True)
    gen = TvmDataGen(x_train, y_train)
    x_batch, y_batch = gen[0]
    pass

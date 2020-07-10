from tensorflow import keras
import os

import data


def train(model: keras.Model):
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, \
        ReduceLROnPlateau
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    weight_path = 'weights/%s.h5' % model.name
    if os.path.exists(weight_path):
        model.load_weights(weight_path, by_name=True)
    x_train, y_train = data.load_train('cifar10', channel_first=False)
    train_iter = data.get_train_iterator(x_train, y_train)
    callbacks = [
        ReduceLROnPlateau(),
        ModelCheckpoint(weight_path, verbose=1, save_best_only=True,
                        save_weights_only=True),
        TensorBoard(),
    ]
    model.fit(train_iter, epochs=100, callbacks=callbacks)


if __name__ == '__main__':
    from resnet.model import get_model
    resnet = get_model()
    train(resnet)

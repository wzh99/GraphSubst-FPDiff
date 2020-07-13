from tensorflow import keras

import common
import data


def train(model: keras.Model):
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, \
        ReduceLROnPlateau
    from tensorflow.keras.optimizers import SGD
    model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    weight_path = 'weights/%s.h5' % model.name
    x_train, y_train = data.load_train('cifar10', channel_first=False)
    train_iter, val_iter = data.get_train_val_iterator(x_train, y_train)
    callbacks = [
        ReduceLROnPlateau(patience=10, min_lr=1e-3, verbose=1),
        ModelCheckpoint(weight_path, verbose=1, save_best_only=True,
                        save_weights_only=True),
        TensorBoard(),
    ]
    steps_per_epoch = int(len(x_train) * (1 - data.val_split)) // common.batch_size
    model.fit(train_iter, epochs=200, callbacks=callbacks,
              steps_per_epoch=steps_per_epoch, validation_data=val_iter)


def test(model: keras.Model):
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    x_test, y_test = data.load_test('cifar10', channel_first=False)
    test_iter = data.get_test_iterator(x_test, y_test)
    model.evaluate(test_iter, batch_size=common.batch_size,
                   steps=len(x_test) // common.batch_size)


if __name__ == '__main__':
    from resnet.model import get_model

    keras_model = get_model(num_stacked=3, load_weights=True)
    # train(keras_model)
    test(keras_model)
    keras.applications.NASNetMobile()

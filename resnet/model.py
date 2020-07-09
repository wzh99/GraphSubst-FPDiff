import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import batch_shape_nhwc, dtype, bn_eps

num_stacked = 3


# noinspection PyTypeChecker
def get_model() -> keras.Model:
    keras.backend.set_floatx(dtype)
    input_data = layers.Input(batch_input_shape=batch_shape_nhwc, dtype=dtype)
    x = layers.Conv2D(16, 3, padding='same', use_bias=False)(input_data)
    for i in range(num_stacked):
        x = _res_block(x, 16)
    x = _res_block(x, 32, strides=2)
    for i in range(num_stacked - 1):
        x = _res_block(x, 32)
    x = _res_block(x, 64, strides=2)
    for i in range(num_stacked - 1):
        x = _res_block(x, 64)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs=input_data, outputs=x)


# noinspection PyTypeChecker
def _res_block(x: tf.Tensor, filters: int, strides: int = 1) -> tf.Tensor:
    shortcut = layers.Conv2D(filters, 1, strides=strides, use_bias=False)(x)
    shortcut = layers.BatchNormalization(momentum=0.9, epsilon=bn_eps)(shortcut)
    x = layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=bn_eps)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=bn_eps)(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


if __name__ == '__main__':
    from workload import Workload
    from tvm.runtime import ndarray
    import numpy as np
    from common import ctx, batch_shape_nchw
    keras_model = get_model()
    keras_model.summary()
    workload = Workload.from_keras(keras_model)
    print(workload.mod.astext())
    # workload.create_executor()
    # workload(ndarray.array(
    #     np.random.randn(*batch_shape_nchw).astype(dtype), ctx=ctx)
    # )

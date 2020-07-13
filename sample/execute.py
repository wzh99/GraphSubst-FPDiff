import numpy as np
import tvm
from tvm import relay

from common import batch_shape_nhwc
from resnet import model

# Load pretrained model
keras_model = model.get_model(3)
keras_model.load_weights('../weights/resnet.h5')

# Load test image
x_keras = np.random.randn(*batch_shape_nhwc).astype('float32')
x_tvm = x_keras.transpose((0, 3, 1, 2))

# Compile model with Relay
shape_dict = {'input_1': x_tvm.shape}
relay_mod, params = relay.frontend.from_keras(keras_model, shape=shape_dict)
print(relay_mod.astext())
target = 'metal'
ctx = tvm.context(target)
with tvm.transform.PassContext(opt_level=3):
    executor = relay.build_module.create_executor(kind='graph', mod=relay_mod,
                                                  ctx=ctx, target=target)
func = executor.evaluate()
y_tvm = func(tvm.nd.array(x_tvm, ctx=ctx), **params)
top1_tvm = np.argmax(y_tvm.asnumpy())
print(top1_tvm)

# Compare with keras
y_keras = keras_model.predict(x_keras)
top1_keras = np.argmax(y_keras)
print(top1_keras)

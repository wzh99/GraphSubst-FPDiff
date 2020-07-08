import numpy as np
import tvm
from PIL import Image
from tensorflow.keras.applications import resnet50
from tvm import relay

# Load pretrained model
keras_model = resnet50.ResNet50(weights=None, input_shape=(224, 224, 3))

# Load test image
img = Image.open('../download/cat.png').resize((224, 224))
x = np.array(img)[np.newaxis, :].astype(np.float32)
x_keras = resnet50.preprocess_input(x)
x_tvm = x_keras.transpose(0, 3, 1, 2)

# Compile model with Relay
shape_dict = {'input_1': x_tvm.shape}
relay_mod, params = relay.frontend.from_keras(keras_model, shape=shape_dict)
# print(relay_mod.astext(show_meta_data=False))
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

# Input (CIFAR-10)
dtype = 'float16'
image_size = (32, 32)
image_chan = 3
batch_size = 100
input_shape_chw = (image_chan,) + image_size
input_shape_hwc = image_size + (image_chan,)
batch_shape_nchw = (batch_size,) + input_shape_chw
batch_shape_nhwc = (batch_size,) + input_shape_hwc

# Runtime
target = 'metal'

# Layer parameter
bn_eps = 1e-5

import tvm
from tvm import relay
from tvm.relay.testing import resnet
from tvm.contrib import graph_runtime, util
import numpy as np


# Define model in Relay
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
relay_mod, params = resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape)
print(relay_mod.astext(show_meta_data=False))

# Compile
target = 'metal'
with tvm.transform.PassContext(opt_level=4):
    graph, lib, params = relay.build(relay_mod, target=target, target_host='llvm',
                                     params=params)

# Run module
ctx = tvm.context(target)
x = np.random.uniform(low=-1, high=1, size=data_shape).astype(np.float32)
graph_exec = graph_runtime.create(graph, lib, ctx)
graph_exec.run(data=x, **params)
y = graph_exec.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

# Save module
tmp = util.tempdir()
graph_path = tmp.relpath('resnet.json')
lib_path = tmp.relpath('resnet.tar')
params_path = tmp.relpath('resnet.params')
lib.export_library(lib_path)
with open(graph_path, 'w') as fo:
    fo.write(graph)
with open(params_path, 'wb') as fo:
    fo.write(relay.save_param_dict(params))
print(tmp.listdir())

# Load module back
graph = open(graph_path, 'r').read()
lib = tvm.runtime.load_module(lib_path)
params = bytearray(open(params_path, 'rb').read())
graph_exec = graph_runtime.create(graph, lib, ctx)
graph_exec.load_params(params)

# Run and compare
graph_exec.run(data=x)
y_deploy = graph_exec.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
tvm.testing.assert_allclose(y_deploy, y, atol=1e-3)

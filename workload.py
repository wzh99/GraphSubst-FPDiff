from typing import Callable, Dict, Optional

from tensorflow import keras
from tvm import relay, ir, runtime, transform
from util import AlterDType

from common import batch_shape_nchw, dtype, target, ctx


class Workload:
    executor: Optional[relay.build_module.GraphExecutor]
    func: Optional[Callable]

    def __init__(self, mod: ir.IRModule, params: Dict[str, runtime.NDArray]):
        self.mod = transform.Sequential(passes=[
            AlterDType(dtype)
        ])(mod)
        self.params = dict([(key, val.asnumpy()) for key, val in params.items()])
        self.executor = None
        self.func = None

    @staticmethod
    def from_keras(model: keras.Model):
        mod, params = relay.frontend.from_keras(
            model, shape={'input_1': batch_shape_nchw}
        )
        return Workload(mod, params)

    def create_executor(self):
        self.executor = relay.build_module.create_executor(
            kind='graph', mod=self.mod, ctx=ctx, target=target
        )
        self.func = self.executor.evaluate()

    def __getitem__(self, item: str):
        return self.params[item]

    def __call__(self, *args):
        if self.func is None:
            raise RuntimeError('Executor is not created.')
        return self.func(*args, **self.params)

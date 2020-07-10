from typing import Callable, Dict, Optional, Union, List, Tuple

import numpy as np
from tensorflow import keras
from tvm import relay, ir, runtime

import common
import graph
from util import AlterDType


class Workload:
    """
    A workload is an object containing computation graph of a network and its
    parameters.
    """
    executor: Optional[relay.build_module.GraphExecutor]
    func: Optional[Callable]

    def __init__(self, mod: ir.IRModule,
                 params: Dict[str, Union[runtime.NDArray, np.ndarray]],
                 dtype: str = common.dtype):
        """
        Constructor.
        :param mod: ir.IRModule
            Relay IR module defining computation graph of the network.
        :param params: Dict[str, Union[runtime.NDArray, np.ndarray]]
            Mapping from parameter names to values.
            Internally, the values are stored as np.ndarray. runtime.NDArray values
            will be converted to that type.
        :param dtype: str
            Data type of network function and parameters.
        """
        self.mod = AlterDType(dtype)(mod)
        self.params = dict([(key, np.array(val if isinstance(val, np.ndarray)
                                           else val.asnumpy(), dtype=dtype))
                            for key, val in params.items()])
        self.dtype = dtype
        self.executor = None
        self.func = None

    @staticmethod
    def from_keras(model: keras.Model):
        mod, params = relay.frontend.from_keras(
            model, shape={'input_1': common.batch_shape_nchw}
        )
        return Workload(mod, params)

    def create_executor(self):
        self.executor = relay.build_module.create_executor(
            kind='graph', mod=self.mod, ctx=common.ctx, target=common.target
        )
        self.func = self.executor.evaluate()

    def as_type(self, dtype: str):
        return Workload(self.mod, self.params, dtype=dtype)

    def __getitem__(self, item: str):
        return self.params[item]

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        if self.func is None:
            raise RuntimeError('Executor is not created.')
        return self.func(*args, **self.params).asnumpy()


class Intermediate:
    """
    Intermediate results of a workload
    """

    def __init__(self, workload: Workload, pattern: relay.Expr):
        """
        Constructor.
        :param workload: Workload
            The workload object whose intermediate results should be retrieved.
        :param pattern: relay.Expr
            Expression pattern for finding breakpoints.
        """
        # Transfer fields from workload
        self.mod = workload.mod
        self.params = workload.params
        self.dtype = workload.dtype

        # Find breakpoints with given pattern
        visitor = _BreakpointVisitor(pattern)
        visitor.visit(self.mod['main'])
        self.interm_mods = [ir.IRModule(functions={
            'main': relay.Function(relay.analysis.free_vars(expr), expr)
        }) for expr in visitor.matched]
        self.interm_mods.append(self.mod)

        # Create workloads for intermediate modules
        self.workloads = [Workload(mod, self.params, dtype=self.dtype)
                          for mod in self.interm_mods]
        for wl in self.workloads:
            wl.create_executor()

    def __call__(self, *args: np.ndarray) -> List[np.ndarray]:
        return [wl(*args) for wl in self.workloads]


class _BreakpointVisitor(relay.ExprVisitor):
    def __init__(self, pattern: relay.Expr):
        super().__init__()
        self.matched: List[relay.Expr] = []
        self.pattern = pattern

    def visit_call(self, call: relay.Call):
        super().visit_call(call)
        if graph.match(self.pattern, call):
            self.matched.append(call)

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        super().visit_tuple_getitem(getitem)
        if graph.match(self.pattern, getitem):
            self.matched.append(getitem)


class DiffCmp:
    """
    Compare differences of intermediate output.
    """

    def __init__(self, fst: Intermediate, snd: Intermediate):
        assert len(fst.interm_mods) == len(snd.interm_mods)
        self.out_len = len(fst.interm_mods)
        self.fst = fst
        self.snd = snd
        self.max = np.ndarray((0, self.out_len), dtype='float32')
        self.mean = np.ndarray((0, self.out_len), dtype='float32')

    def __call__(self, *args: np.ndarray):
        """
        Run one batch on two intermediate objects and add one row to both maximum
        and mean statistical matrices.
        :param args: Tuple[np.ndarray, ...]
            Input tensors.
        """
        out_fst = self.fst(*args)
        out_snd = self.snd(*args)
        diff_max = [np.max(np.abs(y_fst - y_snd))
                    for y_fst, y_snd in zip(out_fst, out_snd)]
        self.max = np.concatenate([self.max, [diff_max]], axis=0)
        diff_mean = [np.mean(np.abs(y_fst - y_snd))
                     for y_fst, y_snd in zip(out_fst, out_snd)]
        self.mean = np.concatenate([self.mean, [diff_mean]], axis=0)

    def report(self):
        """
        Report reduced statistics of all batches.
        """
        np.set_printoptions(formatter={'float': '{:.2e}'.format})
        print('---Statistics for breakpoints---')
        all_max = np.max(self.max, axis=0, keepdims=False)
        print('Maximum:\n', all_max)
        all_mean = np.mean(self.mean, axis=0, keepdims=False)
        print('Mean:\n', all_mean)

from typing import Callable, Dict, Optional, Union, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import trange
from tvm import relay, ir, runtime, transform

import common
import data
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
        self.params = dict([(key, self._cvt_param(val, dtype))
                            for key, val in params.items()])
        self.dtype = dtype
        self.executor = None
        self.func = None

    @staticmethod
    def _cvt_param(x: Union[runtime.NDArray, np.ndarray], dtype: str) -> np.ndarray:
        if isinstance(x, runtime.NDArray):
            x = x.asnumpy()
        if x.dtype.name != dtype:
            x = x.astype(dtype)
        return x

    @staticmethod
    def from_keras(model: keras.Model, dtype: str = 'float32'):
        """
        Build workload from a Keras model.
        :param model: keras.Model
             A Keras model to be converted.
        :param dtype: str
            Data type of target workload.
        :return: Workload
            The built workload object.
        """
        mod, params = relay.frontend.from_keras(
            model, shape={'input_1': common.batch_shape_nchw}
        )
        return Workload(mod, params, dtype=dtype)

    def build(self):
        with transform.PassContext(opt_level=0):
            self.executor = relay.build_module.create_executor(
                kind='graph', mod=self.mod, ctx=runtime.context(common.target),
                target=common.target
            )
        self.func = self.executor.evaluate()

    def as_type(self, dtype: str):
        return Workload(self.mod, self.params, dtype=dtype)

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        if self.func is None:
            raise RuntimeError('Workload has not been built.')
        return self.func(*args, **self.params).asnumpy()

    def evaluate(self, data_gen: data.TvmDataGen):
        num_batches = data_gen.num_batches
        losses = np.ndarray((num_batches,), dtype='float32')
        accuracies = np.ndarray((num_batches,), dtype='float32')
        batch_range = trange(num_batches)
        for i in batch_range:
            x_batch, y_batch = data_gen[i]
            y_pred = self(x_batch)
            losses[i] = keras.losses.sparse_categorical_crossentropy(
                y_batch, y_pred
            ).numpy().mean()
            accuracies[i] = keras.metrics.sparse_categorical_accuracy(
                tf.convert_to_tensor(y_batch),
                tf.convert_to_tensor(y_pred)
            ).numpy().mean()
        print('Loss:', losses.mean(), 'Accuracy:', accuracies.mean())


class IntermRecord:
    """
    Record intermediate results of a workload
    """

    def __init__(self, workload: Workload, patterns: List[relay.Expr]):
        """
        Constructor.
        :param workload: Workload
            The workload object whose intermediate results should be retrieved.
        :param patterns: relay.Expr
            Expression pattern for finding breakpoints.
        """

        # Find breakpoints with given pattern
        self.orig_wl = workload
        visitor = _BreakpointVisitor(patterns)
        visitor.visit(workload.mod['main'])
        interm_mods = [ir.IRModule(functions={
            'main': relay.Function(relay.analysis.free_vars(expr), expr)
        }) for expr in visitor.matched]
        interm_mods.append(workload.mod)

        # Create workloads for intermediate modules
        self.interm_wl = [Workload(mod, workload.params, dtype=workload.dtype)
                          for mod in interm_mods]
        for wl in self.interm_wl:
            wl.build()

    def __call__(self, *args: np.ndarray) -> List[np.ndarray]:
        return [wl(*args) for wl in self.interm_wl]


class _BreakpointVisitor(relay.ExprVisitor):
    def __init__(self, patterns: List[relay.Expr]):
        super().__init__()
        self.matched: List[relay.Expr] = []
        self.patterns = patterns

    def visit_call(self, call: relay.Call):
        super().visit_call(call)
        if graph.match_any(self.patterns, call):
            self.matched.append(call)

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        super().visit_tuple_getitem(getitem)
        if graph.match_any(self.patterns, getitem):
            self.matched.append(getitem)


class IntermCmp:
    """
    Compare differences of intermediate output.
    """

    def __init__(self, fst: IntermRecord, snd: IntermRecord):
        assert len(fst.interm_wl) == len(snd.interm_wl)
        self.out_len = len(fst.interm_wl)
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

    def test(self, data_gen: data.TvmDataGen, ratio: float = 1):
        for i in trange(int(data_gen.num_batches * ratio)):
            x_batch, _ = data_gen[i]
            self(x_batch)

    def report(self):
        """
        Report reduced statistics of all batches.
        """
        np.set_printoptions(formatter={'float': '{:.2e}'.format})
        print('---Differences of breakpoints---')
        all_max = np.max(self.max, axis=0, keepdims=False)
        print('Maximum:\n', all_max)
        all_mean = np.mean(self.mean, axis=0, keepdims=False)
        print('Mean:\n', all_mean)


def compare_two_workloads(fst_wl: Workload, snd_wl: Workload,
                          fst_pat: List[relay.Expr], snd_pat: List[relay.Expr],
                          data_gen: data.TvmDataGen, cmp_ratio: float = 1):
    """
    Compare evaluation and intermediate results of two workloads
    :param fst_wl: Workload
        The first workload.
    :param snd_wl: Workload
        The second workload.
    :param fst_pat: relay.Expr
        Breakpoint pattern of first workload.
    :param snd_pat: relay.Expr
        Breakpoint pattern of second workload.
    :param data_gen: TvmDataGen
        TVM data generator for testing.
    :param cmp_ratio: float
        Ratio of input data for comparing intermediate results.
    """
    # Build and evaluate two workloads
    fst_wl.build()
    print('Evaluating first workload...')
    fst_wl.evaluate(data_gen)
    snd_wl.build()
    print('Evaluating second workload...')
    snd_wl.evaluate(data_gen)

    # Compare intermediate results
    print('Comparing intermediate results...')
    fst_interm = IntermRecord(fst_wl, fst_pat)
    snd_interm = IntermRecord(snd_wl, snd_pat)
    cmp = IntermCmp(fst_interm, snd_interm)
    cmp.test(data_gen, cmp_ratio)
    cmp.report()

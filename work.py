from typing import Dict, Optional, Union, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import trange
from tvm import relay, ir, runtime, transform
from tvm.contrib import graph_runtime
from tvm.relay import dataflow_pattern as dfp

import common
import data
from util import AlterDType


class Workload:
    """
    A workload is an object containing computation graph of a network and its
    parameters.
    """
    executor: Optional[graph_runtime.GraphModule]

    def __init__(self, mod: ir.IRModule,
                 params: Dict[str, Union[runtime.NDArray, np.ndarray]],
                 dtype: str = common.dtype, name: str = ''):
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
        self.mod = relay.transform.InferType()(self.mod)
        self.params = dict([(key, self._cvt_param(val, dtype))
                            for key, val in params.items()])
        self.dtype = dtype
        self.name = name
        self.executor = None

    @staticmethod
    def _cvt_param(x: Union[runtime.NDArray, np.ndarray], dtype: str) -> np.ndarray:
        if isinstance(x, runtime.NDArray):
            x = x.asnumpy()
        if x.dtype.name != dtype:
            x = x.astype(dtype)
        return x

    @staticmethod
    def from_keras(model: keras.Model, dtype: str = common.dtype):
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
        return Workload(mod, params, dtype=dtype, name=model.name)

    def build(self):
        with transform.PassContext(opt_level=0,
                                   config={'tir.disable_vectorize': True}):
            graph, lib, params = relay.build(self.mod, target=common.target,
                                             params=self.params)
        self.executor = graph_runtime.create(
            graph, lib, ctx=runtime.context(common.target)
        )
        self.executor.set_input(**params)

    def as_type(self, dtype: str):
        return Workload(self.mod, self.params, dtype=dtype, name=self.name)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.executor is None:
            self.build()
        self.executor.run(input_1=x)
        return self.executor.get_output(0).asnumpy()

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


class BreakpointRecord:
    """
    Record intermediate results of a workload
    """

    def __init__(self, workload: Workload, pat_list: List[dfp.DFPattern]):
        """
        Constructor.
        :param workload: Workload
            The workload object whose intermediate results should be retrieved.
        :param pat_list: relay.Expr
            List of patterns for finding breakpoints.
        """

        # Find breakpoints with given pattern
        self.orig_wl = workload
        visitor = _BreakpointVisitor(pat_list)
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

    def __call__(self, x: np.ndarray) -> List[np.ndarray]:
        return [wl(x) for wl in self.interm_wl]


class _BreakpointVisitor(relay.ExprVisitor):
    def __init__(self, pat_list: List[dfp.DFPattern]):
        super().__init__()
        self.matched: List[relay.Expr] = []
        self.pat_list = pat_list

    def visit(self, expr: relay.Expr):
        super(_BreakpointVisitor, self).visit(expr)
        if any([pat.match(expr) for pat in self.pat_list]) and \
                not any([expr.same_as(prev) for prev in self.matched]):
            self.matched.append(expr)


class RecordCompare:
    """
    Compare differences of breakpoint output records.
    """

    def __init__(self, fst: BreakpointRecord, snd: BreakpointRecord):
        assert len(fst.interm_wl) == len(snd.interm_wl)
        self.out_len = len(fst.interm_wl)
        self.fst = fst
        self.snd = snd
        self.max = np.ndarray((0, self.out_len), dtype='float32')
        self.mean = np.ndarray((0, self.out_len), dtype='float32')

    def __call__(self, x: np.ndarray):
        """
        Run one batch on two intermediate objects and add one row to both maximum
        and mean statistical matrices.
        :param x: np.ndarray
            Input tensors.
        """
        out_fst = self.fst(x)
        out_snd = self.snd(x)
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
                          fst_pat: List[dfp.DFPattern], snd_pat: List[dfp.DFPattern],
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
    snd_wl.build()
    print('Evaluating first workload...')
    fst_wl.evaluate(data_gen)
    print('Evaluating second workload...')
    snd_wl.evaluate(data_gen)

    # Compare intermediate results
    print('Comparing intermediate results...')
    fst_rcd = BreakpointRecord(fst_wl, fst_pat)
    snd_rcd = BreakpointRecord(snd_wl, snd_pat)
    cmp = RecordCompare(fst_rcd, snd_rcd)
    cmp.test(data_gen, cmp_ratio)
    cmp.report()

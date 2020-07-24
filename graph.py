import copy
from typing import Dict, Type, Optional

import numpy as np
from graphviz import Digraph
from tvm import relay, ir, transform
from tvm.relay import dataflow_pattern as dfp

from work import Workload


class GraphSubst(dfp.DFPatternCallback):
    pattern: dfp.DFPattern

    def __init__(self, params: Dict[str, np.ndarray]):
        super(GraphSubst, self).__init__()
        self.params = params
        self.next_idx = 1

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.Map) \
            -> relay.Expr:
        pass

    def __call__(self, expr: relay.Expr) -> relay.Expr:
        """
        Perform substitution on a matched subgraph.
        The substituted graph is returned. Mutated parameters are stored in the
        object.
        :param expr: relay.Expr
            a subgraph that matches the pattern
        :return: relay.Expr
        """
        new_expr = self.rewrite(expr)
        return new_expr

    def add_var_with_param(self, param: np.ndarray) -> relay.Var:
        var = relay.var(self.next_param_name(), shape=param.shape,
                        dtype=param.dtype.name)
        self[var] = param
        return var

    def next_param_name(self) -> str:
        self._update_next_idx()
        return self._fmt_param_name(self.next_idx)

    def _update_next_idx(self):
        while self.params.__contains__(self._fmt_param_name(self.next_idx)):
            self.next_idx += 1

    @staticmethod
    def _fmt_param_name(idx: int):
        return "_param_%d" % idx

    def __getitem__(self, var: relay.Var) -> np.ndarray:
        return self.params[var.name_hint]

    def __setitem__(self, key: relay.Var, value: np.ndarray):
        self.params[key.name_hint] = value

    def __delitem__(self, key: relay.Var):
        del self.params[key.name_hint]


class SubstPass:
    """
    A pass on workload (IR module and parameters).
    """

    def __init__(self, ctor: Type[GraphSubst]):
        self.ctor = ctor

    def __call__(self, workload: Workload) -> Workload:
        subst = self.ctor(copy.deepcopy(workload.params))
        new_mod = _SubstFuncPass(subst)(workload.mod)
        param_names = [param.name_hint for param in new_mod['main'].params]
        used_params = {}
        for key, val in subst.params.items():
            if param_names.__contains__(key):
                used_params[key] = val
        return Workload(new_mod, used_params, dtype=workload.dtype,
                        name=workload.name)


@relay.transform.function_pass(opt_level=0)
class _SubstFuncPass:
    def __init__(self, subst: GraphSubst):
        self.subst = subst

    def transform_function(self, fn: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        new_body = self.subst.rewrite(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body,
                              ret_type=fn.ret_type)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


def visualize(wl: Workload, name: Optional[str] = None, path: str = ''):
    if name is None:
        name = wl.name
    graph = Digraph(name=name)
    _GraphVizVisitor(graph).visit_function(wl.mod['main'])
    graph.view(directory=path)


class _GraphVizVisitor(relay.ExprVisitor):
    def __init__(self, graph: Digraph):
        super().__init__()
        self.graph = graph
        self.node_id: Dict[relay.Expr, str] = {}
        self.counter = 0

    def visit(self, expr):
        if self.node_id.__contains__(expr):
            return
        super().visit(expr)

    def visit_var(self, var: relay.Var):
        expr_id = self._register_node(var)
        self.graph.node(expr_id, label=var.name_hint)

    def visit_constant(self, const: relay.Constant):
        expr_id = self._register_node(const)
        self.graph.node(expr_id, label='const')

    def visit_call(self, call: relay.Call):
        expr_id = self._register_node(call)
        self.graph.node(expr_id, label=call.op.name)
        for arg in call.args:
            self.visit(arg)
            self.graph.edge(self.node_id[arg], expr_id)

    def visit_tuple(self, tup: relay.Tuple):
        expr_id = self._register_node(tup)
        self.graph.node(expr_id, label='(,)')
        for field in tup.fields:
            self.visit(field)
            self.graph.edge(self.node_id[field], expr_id)

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        expr_id = self._register_node(getitem)
        self.graph.node(expr_id, label='.%d' % getitem.index)
        self.visit(getitem.tuple_value)
        self.graph.edge(self.node_id[getitem.tuple_value], expr_id)

    def _register_node(self, expr: relay.Expr) -> str:
        cur_id = str(self.counter)
        self.node_id[expr] = cur_id
        self.counter += 1
        return cur_id

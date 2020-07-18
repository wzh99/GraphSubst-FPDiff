import copy
from typing import Dict, Type, List

import numpy as np
from tvm import relay, ir, transform

from work import Workload


class GraphSubst:
    def __init__(self, params: Dict[str, np.ndarray]):
        self.params = params
        self.next_idx = 1

    def get_pattern(self) -> relay.Expr:
        """
        Get the expression pattern for subgraph matching.
        :return: relay.Expr
        """
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
        pass

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
        return Workload(new_mod, used_params, dtype=workload.dtype)


@relay.transform.function_pass(opt_level=0)
class _SubstFuncPass:
    def __init__(self, subst: GraphSubst):
        self.subst = subst

    def transform_function(self, func: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        return _SubstMutator(self.subst).visit(func)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


class _SubstMutator(relay.ExprMutator):
    def __init__(self, subst: GraphSubst):
        super().__init__()
        self.subst = subst
        self.pattern = subst.get_pattern()

    def visit_function(self, fn: relay.Function) -> relay.Function:
        new_body = self.visit(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body,
                              ret_type=fn.ret_type)

    def visit_call(self, call: relay.Call) -> relay.Expr:
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, attrs=call.attrs)
        if not match_one(self.pattern, call):
            return new_call
        return self.subst(new_call)

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem) -> relay.Expr:
        new_tuple = self.visit(getitem.tuple_value)
        new_getitem = relay.TupleGetItem(new_tuple, getitem.index)
        if not match_one(self.pattern, new_getitem):
            return new_getitem
        return self.subst(new_getitem)

    def visit_tuple(self, tup: relay.Tuple) -> relay.Expr:
        new_fields = [self.visit(field) for field in tup.fields]
        new_tuple = relay.Tuple(new_fields)
        if not match_one(self.pattern, new_tuple):
            return new_tuple
        return self.subst(new_tuple)


def match_any(pat_list: List[relay.Expr], expr: relay.Expr) -> bool:
    for pat in pat_list:
        if match_one(pat, expr):
            return True
    return False


# noinspection PyTypeChecker
def match_one(pat: relay.Expr, expr: relay.Expr) -> bool:
    """
    Match an expression with a pattern.
    A expression matches another iff. they are of the same type and their
    subexpressions match pairwise. A variable in the pattern matches any
    expression.
    :param pat: relay.Expr
        Expression pattern for matching
    :param expr: relay.Expr
        Candidate expression to be matched
    :return: bool
        Whether `expr` matches `pattern`
    """
    # Check whether node type matches
    if isinstance(pat, relay.Var):
        return True  # a variable matches any expression
    if pat.__class__ != expr.__class__:
        return False

    # Check according to different node type
    if isinstance(pat, relay.Call):
        return _match_call(pat, expr)
    elif isinstance(pat, relay.TupleGetItem):
        return _match_tuple_getitem(pat, expr)
    elif isinstance(pat, relay.Tuple):
        return _match_tuple(pat, expr)
    else:
        return False


def _match_call(pat: relay.Call, expr: relay.Call) -> bool:
    # Match operator
    if pat.op.name != expr.op.name:
        return False

    # Match arity
    if len(pat.args) != len(expr.args):
        return False

    # Match arguments
    return all([match_one(pat_arg, expr_arg)
                for pat_arg, expr_arg in zip(pat.args, expr.args)])


def _match_tuple_getitem(pat: relay.TupleGetItem, expr: relay.TupleGetItem) -> bool:
    return pat.index == expr.index and \
           match_one(pat.tuple_value, expr.tuple_value)


def _match_tuple(pat: relay.Tuple, expr: relay.Tuple) -> bool:
    # Match field numbers
    if len(pat.fields) != len(expr.fields):
        return False

    # Match fields pairwise
    return all([match_one(pat_field, expr_field)
                for pat_field, expr_field in zip(pat.fields, expr.fields)])

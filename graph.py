import copy
from typing import Dict, Type, List

import numpy as np
from tvm import relay, ir, transform

from work import Workload


class GraphSubst:
    def __init__(self, params: Dict[str, np.ndarray]):
        self.params = params
        self.next_idx = 1
        self._update_next_idx()  # skip all preexisting names

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


class WorkloadPass:
    """
    A pass on workload (IR module and parameters).
    """

    def __init__(self, ctor: Type[GraphSubst]):
        self.ctor = ctor

    def __call__(self, workload: Workload) -> Workload:
        subst = self.ctor(copy.deepcopy(workload.params))
        new_mod = _SubstPass(subst)(workload.mod)
        return Workload(new_mod, subst.params)


@relay.transform.function_pass(opt_level=0)
class _SubstPass:
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


def match_any(patterns: List[relay.Expr], expr: relay.Expr) -> bool:
    for pat in patterns:
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
        return True
    if pat.__class__ != expr.__class__:
        return False

    # Check according to different node type
    if isinstance(pat, relay.Call):
        return _match_call(pat, expr)
    elif isinstance(pat, relay.TupleGetItem):
        return _match_tuple_getitem(pat, expr)
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
    for pat_arg, expr_arg in zip(pat.args, expr.args):
        # Don't check further if a variable is encountered
        if isinstance(pat_arg, relay.Var):
            continue
        if not match_one(pat_arg, expr_arg):
            return False

    return True


def _match_tuple_getitem(pat: relay.TupleGetItem, expr: relay.TupleGetItem) -> bool:
    return pat.index == expr.index and \
           match_one(pat.tuple_value, expr.tuple_value)

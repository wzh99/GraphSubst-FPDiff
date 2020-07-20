from tvm import relay, ir, transform


@relay.transform.function_pass(opt_level=0)
class AlterDType:
    """
    Alter data type involved in an expression..
    """

    def __init__(self, tgt_ty: str):
        self.var_mut = _VarDTypeMutator(tgt_ty)

    def transform_function(self, func: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        return self.var_mut.visit(func)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


class _VarDTypeMutator(relay.ExprMutator):
    def __init__(self, tgt_ty: str):
        super().__init__()
        self.ty_mut = _TensorDTypeMutator(tgt_ty)

    def visit_function(self, fn: relay.Function):
        new_func = super().visit_function(fn)
        return relay.Function(new_func.params, new_func.body)

    def visit_var(self, var: relay.Var):
        new_ty = self.ty_mut.visit(var.type_annotation)
        return relay.Var(name_hint=var.name_hint, type_annotation=new_ty)


class _TensorDTypeMutator(relay.TypeMutator):
    def __init__(self, tgt_ty: str):
        super().__init__()
        self.tgt_ty = tgt_ty

    def visit_tensor_type(self, tt: relay.TensorType):
        return relay.TensorType(tt.concrete_shape, dtype=self.tgt_ty)


def infer_type(expr: relay.Expr) -> relay.TensorType:
    """
    Check type of arbitrary expression,
    :param expr: relay.Expr
        The expression whose type will be checked.
    :return: relay.Type
        Type of the expression.
    """
    mod = ir.IRModule(functions={
        'main': relay.Function(relay.analysis.free_vars(expr), expr)
    })
    ty = relay.transform.InferType()(mod)['main'].checked_type.ret_type
    return ty

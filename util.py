from tvm import relay, ir, transform


@relay.transform.function_pass(opt_level=0)
class AlterDType:
    """
    Alter data type involved in an expression..
    """
    def __init__(self, tgt_ty: str):
        self.var_mut = _VarDTypeMutator(tgt_ty)

    def transform_function(self, func: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext):
        return self.var_mut.visit(func)


class _VarDTypeMutator(relay.ExprMutator):
    def __init__(self, tgt_ty: str):
        super().__init__()
        self.ty_mut = _TensorDTypeMutator(tgt_ty)

    def visit_function(self, fn: relay.Function):
        new_func = super().visit_function(fn)
        return relay.Function(new_func.params, new_func.body)

    def visit_var(self, var: relay.Var):
        new_ty = self.ty_mut.visit(var.checked_type)
        return relay.Var(name_hint=var.name_hint, type_annotation=new_ty)


class _TensorDTypeMutator(relay.TypeMutator):
    def __init__(self, tgt_ty: str):
        super().__init__()
        self.tgt_ty = tgt_ty

    def visit_tensor_type(self, tt: relay.TensorType):
        return relay.TensorType(tt.concrete_shape, dtype=self.tgt_ty)

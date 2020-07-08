from tensorflow import keras
from tensorflow.keras import layers
from tvm import relay, ir, transform


@relay.transform.function_pass(opt_level=0)
class AlterDType:
    def __init__(self, tgt_ty: str):
        self.var_mut = VarDTypeMutator(tgt_ty)

    def transform_function(self, func: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext):
        return self.var_mut.visit(func)


class VarDTypeMutator(relay.ExprMutator):
    def __init__(self, tgt_ty: str):
        super().__init__()
        self.ty_mut = TensorDTypeMutator(tgt_ty)

    def visit_function(self, fn: relay.Function):
        new_func = super().visit_function(fn)
        return relay.Function(new_func.params, new_func.body)

    def visit_var(self, var: relay.Var):
        new_ty = self.ty_mut.visit(var.checked_type)
        return relay.Var(name_hint=var.name_hint, type_annotation=new_ty)


class TensorDTypeMutator(relay.TypeMutator):
    def __init__(self, tgt_ty: str):
        super().__init__()
        self.tgt_ty = tgt_ty

    def visit_tensor_type(self, tt: relay.TensorType):
        return relay.TensorType(tt.concrete_shape, dtype=self.tgt_ty)


inputs = keras.Input(shape=(224, 224, 3), batch_size=4)
x = layers.Conv2D(64, 3, use_bias=False, padding='same')(inputs)
x = layers.BatchNormalization(epsilon=1e-5)(x)
x = layers.GlobalAvgPool2D()(x)
keras_model = keras.Model(inputs=inputs, outputs=x)
keras_model.summary()
ir_mod, params = relay.frontend.from_keras(keras_model,
                                           shape={'input_1': (4, 3, 224, 224)})
print(ir_mod)
ir_mod = transform.Sequential(passes=[
    AlterDType('float16'),
    relay.transform.InferType(),
])(ir_mod)
print(ir_mod)

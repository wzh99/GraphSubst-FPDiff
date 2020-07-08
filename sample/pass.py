import tvm
from tvm import relay


@relay.transform.function_pass(opt_level=1)
class ReplaceFuncPass:
    def __init__(self, tgt_func: relay.Function):
        self.tgt_func = tgt_func

    def transform_function(self, func: relay.Function, mod: tvm.ir.IRModule,
                           ctx: tvm.transform.PassContext):
        return self.tgt_func


x = relay.var('x', shape=(10, 20), dtype='float16')
f1 = relay.Function([x], relay.abs(x))
f2 = relay.Function([x], relay.log(x))
func_pass = tvm.transform.Sequential(passes=[
    relay.transform.InferType(),
    ReplaceFuncPass(f2)
])
mod = tvm.IRModule(functions={'main': f1})
# print(mod.astext())
mod = func_pass(mod)
# print(mod.astext())

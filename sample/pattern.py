from tvm import relay
from tvm.relay import dataflow_pattern as dfp


def get_pattern() -> dfp.DFPattern:
    x = dfp.wildcard()
    w = dfp.is_var()
    x = dfp.is_op('nn.conv2d')(x, w)
    x = dfp.is_op('nn.relu')(x)
    return x


def get_expr() -> relay.Expr:
    x = relay.var('x', shape=(16, 3, 32, 32), dtype='float32')
    w = relay.var('w', shape=(32, 3, 3, 3))
    x = relay.nn.conv2d(x, w, padding=(1, 1))
    x = relay.nn.relu(x)
    return x


print(get_pattern().match(get_expr()))

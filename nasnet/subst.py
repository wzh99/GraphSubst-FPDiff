from typing import List

import numpy as np
from tvm import relay

import util
from graph import GraphSubst


class ConvAddSubst(GraphSubst):
    def get_pattern(self) -> relay.Expr:
        x1 = relay.var('x1')
        w1 = relay.var('w1')
        x1 = relay.nn.conv2d(x1, w1)
        b1 = relay.var('b1')
        x1 = relay.nn.bias_add(x1, b1)
        x2 = relay.var('x2')
        w2 = relay.var('w2')
        x2 = relay.nn.conv2d(x2, w2)
        b2 = relay.var('b2')
        x2 = relay.nn.bias_add(x2, b2)
        x = relay.add(x1, x2)
        return x

    def __call__(self, expr: relay.Expr) -> relay.Expr:
        # Extract related expressions
        bias_add_1, bias_add_2 = expr.args
        conv_1, b1_var = bias_add_1.args
        conv_2, b2_var = bias_add_2.args
        x1_var, w1_var = conv_1.args
        x2_var, w2_var = conv_2.args

        # Look up parameters
        b1_param = self[b1_var]
        b2_param = self[b2_var]
        w1_param = self[w1_var]
        w2_param = self[w2_var]

        # Fuse parameters
        fused_w_param = np.concatenate([w1_param, w2_param], axis=1)
        fused_w_var = self.add_var_with_param(fused_w_param)
        fused_b_param = b1_param + b2_param
        fused_b_var = self.add_var_with_param(fused_b_param)

        # Rebuild subgraph
        x = relay.concatenate([x1_var, x2_var], 1)
        x = relay.nn.conv2d(x, fused_w_var)
        x = relay.nn.bias_add(x, fused_b_var)

        return x


class AvgAddSubst(GraphSubst):
    def get_pattern(self) -> relay.Expr:
        x = relay.var('x')
        avg1 = relay.nn.avg_pool2d(x, pool_size=(3, 3), padding=(1, 1))
        avg2 = relay.nn.avg_pool2d(x, pool_size=(3, 3), padding=(1, 1))
        x = relay.add(avg1, avg2)
        return x

    def __call__(self, add: relay.Expr) -> relay.Expr:
        avg1, avg2 = add.args
        x = avg1.args[0]
        x_type = util.infer_type(x)
        num_feat = int(x_type.shape[1])
        weight = (2. / 9) * np.ones((num_feat, 1, 3, 3), dtype=x_type.dtype)
        x = relay.nn.conv2d(x, relay.const(weight, dtype=x_type.dtype),
                            padding=(1, 1), groups=num_feat)
        return x


def _get_breakpoint_patterns() -> List[relay.Expr]:
    norm_pat = relay.concatenate([
        relay.var('b1'), relay.var('b2'), relay.var('b3'),
        relay.var('b4'), relay.var('b5'), relay.var('b6')
    ], 1)
    red_pat = relay.concatenate([
        relay.var('b1'), relay.var('b2'), relay.var('b3'), relay.var('b4')
    ], 1)
    return [norm_pat, red_pat]


if __name__ == '__main__':
    from nasnet import get_model
    from resnet import ConvBnSubst
    from graph import SubstPass
    import work
    import data

    x_test, y_test = data.load_test('cifar10', channel_first=True)
    test_gen = data.TvmDataGen(x_test, y_test)
    nasnet = get_model(6, load_weights=True)
    wl_1 = work.Workload.from_keras(nasnet, dtype='float16')
    wl_2 = SubstPass(ConvBnSubst)(wl_1)
    wl_3 = SubstPass(ConvAddSubst)(wl_2)
    wl_4 = SubstPass(AvgAddSubst)(wl_3)
    # wl_4.evaluate(test_gen)
    # wl_1.evaluate(test_gen)
    pat_list = _get_breakpoint_patterns()
    rcd_1 = work.BreakpointRecord(wl_1, pat_list)
    rcd_4 = work.BreakpointRecord(wl_4, pat_list)
    # cmp = work.RecordCompare(rcd_1, rcd_4)
    # cmp.test(test_gen, 0.05)
    # cmp.report()
    pass

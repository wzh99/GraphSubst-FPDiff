from typing import List, Dict

import numpy as np
from tvm import relay, ir
from tvm.relay import dataflow_pattern as dfp

import util
from graph import GraphSubst


class ConvAddSubst(GraphSubst):
    def __init__(self, params: Dict[str, np.ndarray]):
        super(ConvAddSubst, self).__init__(params)

        self.x1 = dfp.wildcard()
        self.w1 = dfp.is_var()
        x1 = dfp.is_op('nn.conv2d')(self.x1, self.w1)
        self.b1 = dfp.is_var()
        x1 = dfp.is_op('nn.bias_add')(x1, self.b1)
        self.x2 = dfp.wildcard()
        self.w2 = dfp.is_var()
        x2 = dfp.is_op('nn.conv2d')(self.x2, self.w2)
        self.b2 = dfp.is_var()
        x2 = dfp.is_op('nn.bias_add')(x2, self.b2)
        x = x1 + x2
        self.pattern = x

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.Map) \
            -> relay.Expr:
        # Extract variables
        x1 = node_map[self.x1][0]
        x2 = node_map[self.x2][0]
        b1_var = node_map[self.b1][0]
        b2_var = node_map[self.b2][0]
        w1_var = node_map[self.w1][0]
        w2_var = node_map[self.w2][0]

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
        x = relay.concatenate([x1, x2], 1)
        x = relay.nn.conv2d(x, fused_w_var)
        x = relay.nn.bias_add(x, fused_b_var)

        return x


class AvgAddSubst(GraphSubst):
    def __init__(self, params: Dict[str, np.ndarray]):
        super(AvgAddSubst, self).__init__(params)

        self.x = dfp.wildcard()
        avg1 = dfp.is_op('nn.avg_pool2d')(self.x)
        avg2 = dfp.is_op('nn.avg_pool2d')(self.x)
        x = avg1 + avg2
        self.pattern = x

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.Map) \
            -> relay.Expr:
        x = node_map[self.x][0]
        x_type = util.infer_type(x)
        num_feat = int(x_type.shape[1])
        weight = (2. / 9) * np.ones((num_feat, 1, 3, 3), dtype=x_type.dtype)
        x = relay.nn.conv2d(x, relay.const(weight, dtype=x_type.dtype),
                            padding=(1, 1), groups=num_feat)
        return x


# noinspection PyTypeChecker
def _get_breakpoint_patterns() -> List[dfp.DFPattern]:
    norm = dfp.is_op('concatenate')(
        dfp.is_tuple((
            dfp.wildcard(), dfp.wildcard(), dfp.wildcard(),
            dfp.wildcard(), dfp.wildcard(), dfp.wildcard()
        ))
    )
    red = dfp.is_op('concatenate')(
        dfp.is_tuple((
            dfp.wildcard(), dfp.wildcard(), dfp.wildcard(), dfp.wildcard()
        ))
    )
    return [norm, red]


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
    wl_4.evaluate(test_gen)
    # wl_1.evaluate(test_gen)
    # pat_list = _get_breakpoint_patterns()
    # rcd_1 = work.BreakpointRecord(wl_1, pat_list)
    # rcd_4 = work.BreakpointRecord(wl_4, pat_list)
    # cmp = work.RecordCompare(rcd_1, rcd_4)
    # cmp.test(test_gen, 0.05)
    # cmp.report()
    pass

from typing import List, Dict

import numpy as np
from tvm import relay, ir
from tvm.relay import dataflow_pattern as dfp

from common import bn_eps
from graph import GraphSubst


class ConvBnSubst(GraphSubst):
    def __init__(self, params: Dict[str, np.ndarray]):
        super(ConvBnSubst, self).__init__(params)

        # Setup pattern
        self.x = dfp.wildcard()
        self.weight = dfp.is_var()
        self.conv = dfp.is_op('nn.conv2d')(self.x, self.weight)
        self.gamma = dfp.is_var()
        self.beta = dfp.is_var()
        self.moving_mean = dfp.is_var()
        self.moving_var = dfp.is_var()
        x = dfp.is_op('nn.batch_norm')(self.conv, self.gamma, self.beta,
                                       self.moving_mean, self.moving_var)
        x = dfp.is_tuple_get_item(x, 0)
        self.pattern = x

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.Map) \
            -> relay.Expr:
        # Extract variables from node map
        x = node_map[self.x][0]
        weight_var = node_map[self.weight][0]
        gamma_var = node_map[self.gamma][0]
        beta_var = node_map[self.beta][0]
        moving_mean_var = node_map[self.moving_mean][0]
        moving_var_var = node_map[self.moving_var][0]
        conv_call = node_map[self.conv][0]

        # Get parameters for variables involved
        weight_param = self[weight_var]
        gamma_param = self[gamma_var]
        beta_param = self[beta_var]
        moving_mean_param = self[moving_mean_var]
        moving_var_param = self[moving_var_var]

        # Fuse parameters
        conv_weight_mat = weight_param.reshape((weight_param.shape[0], -1))
        bn_weight = np.diag(gamma_param / np.sqrt(moving_var_param + bn_eps))
        fused_weight_param = np.matmul(bn_weight, conv_weight_mat) \
            .reshape(weight_param.shape)
        fused_bias_param = beta_param - gamma_param * moving_mean_param / np.sqrt(
            moving_var_param + bn_eps)
        fused_weight_var = self.add_var_with_param(fused_weight_param)
        fused_bias_var = self.add_var_with_param(fused_bias_param)

        # Reconstruct subgraph
        x = relay.Call(conv_call.op, [x, fused_weight_var], attrs=conv_call.attrs)
        x = relay.nn.bias_add(x, fused_bias_var)

        return x


def _get_breakpoint_patterns() -> List[dfp.DFPattern]:
    x = dfp.wildcard()
    shortcut = dfp.wildcard()
    x = dfp.is_op('add')(x, shortcut)
    x = dfp.is_op('nn.relu')(x)
    return [x]


if __name__ == '__main__':
    import work
    from graph import SubstPass
    from resnet import get_model
    import data

    resnet = get_model(3, load_weights=True)
    x_test, y_test = data.load_test('cifar10', channel_first=True)
    test_gen = data.TvmDataGen(x_test, y_test)
    wl = work.Workload.from_keras(resnet, dtype='float16')
    subst_wl = SubstPass(ConvBnSubst)(wl)
    # wl = wl.as_type(common.dtype)
    # subst_wl = subst_wl.as_type(common.dtype)
    pat_list = _get_breakpoint_patterns()
    work.compare_two_workloads(wl, subst_wl, pat_list, pat_list, test_gen, 0.1)

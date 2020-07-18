import numpy as np
from tvm import relay

from common import bn_eps
from graph import GraphSubst


class ConvBnSubst(GraphSubst):
    def get_pattern(self) -> relay.Expr:
        x = relay.var('x')
        weight = relay.var('conv_weight')
        gamma = relay.var('bn_gamma')
        beta = relay.var('bn_beta')
        moving_mean = relay.var('bn_moving_mean')
        moving_var = relay.var('bn_moving_var')
        x = relay.nn.conv2d(x, weight)
        x, _, _ = relay.nn.batch_norm(x, gamma, beta, moving_mean, moving_var)
        return x

    def __call__(self, expr: relay.TupleGetItem) -> relay.Expr:
        # Extract related expressions and variables
        bn_call = expr.tuple_value
        conv_call, gamma_var, beta_var, moving_mean_var, moving_var_var = bn_call.args
        x, weight_var = conv_call.args

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


def _get_breakpoint_pattern() -> relay.Expr:
    s = relay.var('s')
    x = relay.var('x')
    x = relay.add(x, s)
    x = relay.nn.relu(x)
    return x


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
    pat = _get_breakpoint_pattern()
    work.compare_two_workloads(wl, subst_wl, [pat], [pat], test_gen, 0.1)

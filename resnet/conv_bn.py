from typing import Tuple

import numpy as np
from tvm import relay

import common
from graph import GraphSubst

# Parameters used in pattern expression
dtype = common.dtype
pat_input_chan = common.image_chan
pat_batch_shape = common.batch_shape_nchw
pat_num_feat = 64
pat_kernel_size = 3
pat_weight_shape = (pat_num_feat, pat_input_chan, pat_kernel_size, pat_kernel_size)
eps = common.bn_eps


def get_pattern() -> relay.Expr:
    """
    Get pattern of a source convolution-BN subgraph to be substituted for.
    :return: relay.Expr
     With the following free variables:
        %x: input features
        %conv_weight: weights of convolution
        %bn_gamma: scale parameter of BN
        %bn_beta: shift parameter of BN
        %bn_moving_mean: mean estimate of BN
        %bn_moving_var: variance estimate of BN
    """
    x = relay.var('x', shape=pat_batch_shape, dtype=dtype)
    weight = relay.var('conv_weight', shape=pat_weight_shape, dtype=dtype)
    gamma = relay.var('bn_gamma', shape=(pat_num_feat,), dtype=dtype)
    beta = relay.var('bn_beta', shape=(pat_num_feat,), dtype=dtype)
    moving_mean = relay.var('bn_moving_mean', shape=(pat_num_feat,), dtype=dtype)
    moving_var = relay.var('bn_moving_var', shape=(pat_num_feat,), dtype=dtype)
    x = relay.nn.conv2d(x, weight, padding=(1, 1))
    x, _, _ = relay.nn.batch_norm(x, gamma, beta, moving_mean, moving_var)
    return x


def fuse_params(conv_weight: np.ndarray, bn_gamma: np.ndarray, bn_beta: np.ndarray,
                bn_moving_mean: np.ndarray, bn_moving_var: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse parameters of convolution-BN subgraph
    :return: Tuple[np.ndarray, np.ndarray]
        [0]: fused weight
        [1]: fused bias
    """
    conv_weight_mat = conv_weight.reshape((conv_weight.shape[0], -1))
    bn_weight = np.diag(bn_gamma / np.sqrt(bn_moving_var + eps))
    fused_weight = np.matmul(bn_weight, conv_weight_mat).reshape(conv_weight.shape)
    fused_bias = bn_beta - bn_gamma * bn_moving_mean / np.sqrt(bn_moving_var + eps)
    return fused_weight, fused_bias


class ConvBnSubst(GraphSubst):
    def get_pattern(self) -> relay.Expr:
        return get_pattern()

    def __call__(self, expr: relay.TupleGetItem) -> relay.Expr:
        # Extract related expressions and variables
        bn_call = expr.tuple_value
        conv_call, bn_gamma_var, bn_beta_var, bn_moving_mean_var, bn_moving_var_var \
            = bn_call.args
        input_expr, conv_weight_var = conv_call.args

        # Get parameters for variables involved
        conv_weight_param = self[conv_weight_var]
        bn_gamma_param = self[bn_gamma_var]
        bn_beta_param = self[bn_beta_var]
        bn_moving_mean_param = self[bn_moving_mean_var]
        bn_moving_var_param = self[bn_moving_var_var]
        for var in [conv_weight_var, bn_gamma_var, bn_beta_var, bn_moving_mean_var,
                    bn_moving_var_var]:
            del self[var]

        # Fuse parameters
        fused_weight_param, fused_bias_param = fuse_params(
            conv_weight_param, bn_gamma_param, bn_beta_param, bn_moving_mean_param,
            bn_moving_var_param
        )
        fused_weight_var = relay.var(self.next_param_name())
        self[fused_weight_var] = fused_weight_param
        fused_bias_var = relay.var(self.next_param_name())
        self[fused_bias_var] = fused_bias_param

        # Reconstruct subgraph
        new_conv_call = relay.Call(conv_call.op, [input_expr, fused_weight_var],
                                   attrs=conv_call.attrs)
        bias_add = relay.nn.bias_add(new_conv_call, fused_bias_var)

        return bias_add


if __name__ == '__main__':
    from workload import Workload
    from graph import WorkloadPass
    from resnet import model
    keras_model = model.get_model()
    workload = Workload.from_keras(keras_model)
    workload.create_executor()
    new_workload = WorkloadPass(ConvBnSubst)(workload)
    new_workload.create_executor()
    x = np.random.randn(*common.batch_shape_nchw)
    y_orig = workload(x)
    y_new = new_workload(x)
    diff = np.max(np.abs(y_new - y_orig))
    print(diff)

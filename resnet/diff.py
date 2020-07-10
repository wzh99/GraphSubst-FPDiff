import numpy as np
from tvm import relay, ir, transform
from tvm.runtime import ndarray as nd

from common import dtype, target, ctx


def test_conv_bn_subgraph():
    """
    Test difference between a original convolution-BN subgraph and its substituted
    version.
    """
    from resnet import conv_bn as cb

    batch_shape = cb.pat_batch_shape
    num_feat = cb.pat_num_feat
    weight_shape = cb.pat_weight_shape

    # Direct convolution then BN
    pat = cb.get_pattern()
    f = relay.Function(relay.analysis.free_vars(pat), pat)
    orig_mod = ir.IRModule(functions={'main': f})
    orig_mod = relay.transform.InferType()(orig_mod)
    print(orig_mod)

    # Create weights for original module
    conv_weight = np.random.randn(*weight_shape).astype(dtype)
    bn_gamma = np.random.randn(num_feat).astype(dtype)
    bn_beta = np.random.randn(num_feat).astype(dtype)
    bn_moving_var = np.abs(np.random.randn(num_feat)).astype(dtype)
    bn_moving_mean = np.random.randn(num_feat).astype(dtype)
    orig_params = {
        'conv_weight': nd.array(conv_weight, ctx=ctx),
        'bn_gamma': nd.array(bn_gamma, ctx=ctx),
        'bn_beta': nd.array(bn_beta, ctx=ctx),
        'bn_moving_mean': nd.array(bn_moving_mean, ctx=ctx),
        'bn_moving_var': nd.array(bn_moving_var, ctx=ctx)
    }

    # Fused convolution and BN
    x = relay.var('x', shape=batch_shape, dtype=dtype)
    weight = relay.var('weight', shape=weight_shape, dtype=dtype)
    bias = relay.var('bias', shape=(num_feat,), dtype=dtype)
    x = relay.nn.conv2d(x, weight, padding=(1, 1))
    x = relay.nn.bias_add(x, bias)
    f = relay.Function(relay.analysis.free_vars(x), x)
    fused_mod = ir.IRModule(functions={'main': f})
    fused_mod = relay.transform.InferType()(fused_mod)
    print(fused_mod)

    # Create weights for fused module
    fused_weight, fused_bias = cb.fuse_params(conv_weight, bn_gamma, bn_beta,
                                              bn_moving_mean, bn_moving_var)
    fused_params = {
        'weight': nd.array(fused_weight, ctx=ctx),
        'bias': nd.array(fused_bias, ctx=ctx)
    }

    # Compile and run both modules
    input_data = np.random.randn(*batch_shape).astype(dtype)
    with transform.PassContext(opt_level=1):
        orig_exec = relay.build_module \
            .create_executor(kind='graph', mod=orig_mod, ctx=ctx, target=target)
    y_orig = orig_exec.evaluate()(nd.array(input_data, ctx=ctx),
                                  **orig_params).asnumpy()
    with transform.PassContext(opt_level=1):
        fused_exec = relay.build_module \
            .create_executor(kind='graph', mod=fused_mod, ctx=ctx, target=target)
    y_fused = fused_exec.evaluate()(nd.array(input_data, ctx=ctx),
                                    **fused_params).asnumpy()
    print(np.max(np.abs(y_fused - y_orig)))


if __name__ == '__main__':
    test_conv_bn_subgraph()

from graph import GraphSubst
import numpy as np

from tvm import relay
import common

pat_num_feat = 64
pat_batch_shape = (common.batch_size, pat_num_feat) + common.image_size
pat_kernel_size = (1, 1)
pat_weight_shape = (pat_num_feat, pat_num_feat) + pat_kernel_size


class ConvAddSubst(GraphSubst):
    def get_pattern(self) -> relay.Expr:
        x1 = relay.var('x1', shape=pat_batch_shape)
        w1 = relay.var('w1', shape=pat_weight_shape)
        x1 = relay.nn.conv2d(x1, w1)
        b1 = relay.var('b1', shape=(pat_num_feat,))
        x1 = relay.nn.bias_add(x1, b1)
        x2 = relay.var('x2', shape=pat_batch_shape)
        w2 = relay.var('w2', shape=pat_weight_shape)
        x2 = relay.nn.conv2d(x2, w2)
        b2 = relay.var('b2', shape=(pat_num_feat,))
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


if __name__ == '__main__':
    from nasnet import get_model
    from resnet import ConvBnSubst
    from graph import SubstPass
    from work import Workload
    import data

    x_test, y_test = data.load_test('cifar10', channel_first=True)
    test_gen = data.TvmDataGen(x_test, y_test)
    nasnet = get_model(load_weights=True)
    wl_1 = Workload.from_keras(nasnet, dtype='float16')
    wl_2 = SubstPass(ConvBnSubst)(wl_1)
    wl_3 = SubstPass(ConvAddSubst)(wl_2)
    print(np.max(np.abs(wl_3(test_gen[0][0]) - wl_1(test_gen[0][0]))))

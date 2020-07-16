if __name__ == '__main__':
    from nasnet import get_model
    from resnet import ConvBnSubst
    from graph import WorkloadPass
    from work import Workload
    import data

    x_test, y_test = data.load_test('cifar10', channel_first=True)
    test_gen = data.TvmDataGen(x_test, y_test)
    nasnet = get_model(load_weights=True)
    wl = Workload.from_keras(nasnet, dtype='float16')
    subst_wl = WorkloadPass(ConvBnSubst)(wl)
    wl.build()
    wl.evaluate(test_gen)
    # subst_wl.build()
    # subst_wl.evaluate(test_gen)

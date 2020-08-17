# Floating Point Differences of Graph Substitution

## Introduction

Machine learning frameworks perform substitutions on computation graph of models. However, some graph substitutions could cause computational differences, if the model operate on floating point tensors. This project carries out experiments on this. It can evaluate the influence of substitutions on graph and analyze difference of intermediate results.

## Dependency

The project is written in Python 3. Please make sure the following Python packages are installed to use this project:

* tvm 0.7.dev1
* tensorflow >= 2.0
* numpy
* tqdm
* graphviz

## Usage

### Dataset

This project support loading [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) as testing dataset. Feed loaded dataset tensors into a data generator for automatic preprocessing.

```python
import data

x_test, y_test = data.load_test('cifar10', channel_first=True)
test_gen = data.TvmDataGen(x_test, y_test)
```

### Create Workload

In this project, a workload is defined as a computation graph (function) of a model, along with its corresponding parameters. A workload can be created from a Keras model. 

```python
import work

resnet = get_model(3, load_weights=True)
wl_1 = work.Workload.from_keras(resnet, dtype='float16')
```

You can visualize the computation graph of a workload by calling `graph.visualize`:

```python
import graph

graph.visualize(wl_1, name='resnet20', path='logs')
```

### Define Substitutions

A graph substitution is defined in subclasses of `graph.GraphSubst`. The implementation is quite similar to a [dataflow pattern callback](https://tvm.apache.org/docs/langref/relay_pattern.html#applications) in Relay, except that you can also alter parameters. In the constructor, you define patterns to be matched. In overridden `callback` method, you define what the new subgraph will become and how parameters should be altered. See [resnet/subst.py](resnet/subst.py) and [nasnet/subst.py](nasnet/subst.py) for reference. In the following, I will use `ConvBnSubst` as demonstration.

After the graph is defined, you can apply it on a workload. You just need to pass the class name (not the object of this class) to constructor of `graph.SubstPass`, and call it on a workload. 

```python
wl_2 = graph.SubstPass(ConvBnSubst)(wl_1)
```

### Evaluate Workloads

Call `evaluate` method of a workload on a data generator to evaluate. Accuracy and loss values will be computed.

```python
wl_2.evaluate(test_gen)
```

### Compare Intermediate Results

The project capture intermediate results by setting breakpoints on the computation graph. You have to provide a list of patterns for identifying breakpoints. Then breakpoint records should be created for workloads. Finally you create a comparing object and run on data generators. Since this step is resource-consuming, you can specify a ratio to run on only a portion of data. 

```python
def _get_breakpoint_patterns() -> List[dfp.DFPattern]:
    x = dfp.wildcard()
    shortcut = dfp.wildcard()
    x = dfp.is_op('add')(x, shortcut)
    x = dfp.is_op('nn.relu')(x)
    return [x]

pat_list = _get_breakpoint_patterns()
rcd_1 = work.BreakpointRecord(wl_1, pat_list)
rcd_2 = work.BreakpointRecord(wl_2, pat_list)
cmp = work.RecordCompare(rcd_1, rcd_2)
cmp.test(test_gen, 0.03)
cmp.report()
```

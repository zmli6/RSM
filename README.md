# RSM
## Introduction
Subgraph matching is one of the fundamental problems in graph analytics. Existing methods generate matching orders to guide their search, which consists of a series of extensions. Each time, they extend smaller partial matches into larger ones until all complete answers are obtained. However, these methods have two significant drawbacks. Firstly, their matching order generations are usually heuristic and challenging to be effective for different queries. Sec- ondly, each extension, serving as its computation unit, is coarse- grained and may hinder performance. This granularity issue stems from merging generation and expansion operations into a single computation unit. To address these challenges, we introduce a pio- neering framework for Reinforced Subgraph Matching (RSM) that features a fine-grained operation-based search plan. Initially, RSM proposes a fresh paradigm for search , referred to as operation-level search, where each computation unit is defined as an operation that either generates or expands a candidate set under a query vertex. To deal with the second problem and fully exploit the potential of this novel search paradigm, RSM implements a reinforcement learning strategy to generate operation-level search plans. RSM’s reinforce- ment learning approach for constructing operation-based search plans encompasses three modules. In the first module, we employ graph neural networks to extract query vertex representation from graphs. Then, the other two modules leverage multilayer perceptron and are designed to create the generation and expansion operations, respectively. Extensive experiments on real-world graph datasets validate that RSM cuts down query processing time, outperforming existing algorithms by up to 1 to 2 orders of magnitude.


## Input
Both the input query graph and data graph are vertex-labeled.
Each graph starts with 't N M' where N is the number of vertices and M is the number of edges. A vertex and an edge are formatted
as 'v VertexID LabelId Degree' and 'e VertexId VertexId' respectively. Note that we require that the vertex
id is started from 0 and the range is [0,N - 1] where V is the vertex set. The following
is an input sample. You can also find sample data sets and query sets under the test folder.

Example:

```zsh
t 5 6
v 0 0 2
v 1 1 3
v 2 2 3
v 3 1 2
v 4 2 2
e 0 1
e 0 2
e 1 2
e 1 3
e 2 4
e 3 4
```

### Notice
Since the training process may take several hours to several weeks, to facilitate your use, we provide some pre-trained models in the `model_parameter` folder.

Thanks to [Rapids @ HKUST's SubgraphMatching Survey Repo](https://github.com/RapidsAtHKUST/SubgraphMatching) for the implementation of RSM.
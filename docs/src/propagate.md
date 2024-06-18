```@meta
CurrentModule = ModelVerification
```

```@contents
Pages=["propagation.md"]
Depth = 3
```

# Propagation
Functions for propagating the bound through the model (from start nodes to the end nodes) for a given branch. For a forward propagation method (`ForwardProp`), the start nodes are the input nodes of the computational graph and the end nodes are the output nodes. For a backward propagation method (`BackwardProp`), the start nodes are the output nodes and the end nodes are the input nodes. We use BFS (Breadth-first Search) to iterate through the computational graph and propagate the bounds from nodes to nodes.

The `propagate\propagate.jl` module defines algorithms for propagating bounds from input to output, for both forward propagation and backward propagation.

The `propagate\operators` folder contains specific propagation algorithms for different operators, such as ReLU, Dense, Identity, Convolution, Bivariate, etc.

```@docs
propagate
propagate_skip_method(prop_method::ForwardProp, model_info, batch_info, node)
propagate_skip_method(prop_method::BackwardProp, model_info, batch_info, node)
propagate_layer_method(prop_method::ForwardProp, model_info, batch_info, node)
propagate_layer_method(prop_method::BackwardProp, model_info, batch_info, node)
propagate_linear_batch(prop_method::ForwardProp, layer, batch_reach::AbstractArray, batch_info)
propagate_act_batch(prop_method::ForwardProp, σ, batch_reach::AbstractArray, batch_info)
propagate_skip_batch(prop_method::ForwardProp, layer, batch_reach1::AbstractArray, batch_reach2::AbstractArray, batch_info)
is_activation(l)
propagate_layer_batch(prop_method, layer, batch_bound, batch_info)
enqueue_nodes!(prop_method::ForwardProp, queue, model_info)
enqueue_nodes!(prop_method::BackwardProp, queue, model_info)
output_node(prop_method::ForwardProp, model_info)
next_nodes(prop_method::ForwardProp,  model_info, node)
next_nodes(prop_method::BackwardProp, model_info, node)
prev_nodes(prop_method::ForwardProp,  model_info, node)
prev_nodes(prop_method::BackwardProp, model_info, node)
all_nexts_in(prop_method, model_info, output_node, cnt)
all_prevs_in(prop_method, model_info, output_node, cnt)
has_two_reach_node(prop_method, model_info, node)
```

## Bivariate
```@autodocs
Modules=[ModelVerification]
Pages=["bivariate.jl"]
```

## Convolution
```@autodocs
Modules=[ModelVerification]
Pages=["convolution.jl"]
```

## Dense
```@autodocs
Modules=[ModelVerification]
Pages=["dense.jl"]
```

## Identity
```@autodocs
Modules=[ModelVerification]
Pages=["identity.jl"]
```

## Normalise
```@autodocs
Modules=[ModelVerification]
Pages=["normalise.jl"]
```

## ReLU
```@autodocs
Modules=[ModelVerification]
Pages=["relu.jl"]
```

## Stateless
```@autodocs
Modules=[ModelVerification]
Pages=["stateless.jl"]
```
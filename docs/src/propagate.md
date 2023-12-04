```@contents
Pages=["propagation.md"]
Depth = 3
```

# Propagation
Functions for propagating the bound through the model (from start nodes to the end nodes) for a given branch. For a forward propagation method (`ForwardProp`), the start nodes are the input nodes of the computational graph and the end nodes are the output nodes. For a backward propagation method (`BackwardProp`), the start nodes are the output nodes and the end nodes are the input nodes. We use BFS (Breadth-first Search) to iterate through the computational graph and propagates the bounds from nodes to nodes.

The `propagate\propagate.jl` module defines algorithms for propagating bounds from input to output, for both forward propagation and backward propagation.

The `propagate\operators` folder contains specific propagation algorithms for different operators, such as ReLU, Dense, Identity, Convolution, Bivariate, etc.

```@docs
PropMethod
```

```@autodocs
Modules=[ModelVerification]
Pages=["propagate.jl"]
```

NOTE: Need to include `ForwardProp, BackwardProp, ...` from `solvers\solver.jl`.

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
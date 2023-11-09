# NeuralVerification.jl

*A library of algorithms for verifying deep neural networks.
At the moment, all of the algorithms are written under the assumption of feedforward, fully-connected NNs,
and some of them also assume ReLU activation, but both of these assumptions will be relaxed in the future.*

```@contents
Pages = ["index.md", "existing_implementations.md"]
Depth = 2
```

## Installation
To download this library, clone it from the julia package manager like so:
```julia
(v1.0) pkg> add https://github.com/sisl/NeuralVerification.jl
```
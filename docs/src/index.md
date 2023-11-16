# ModelVerification.jl

## Introduction
[ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) is a Julia package for verifying deep neural networks.

```@contents
Pages=["index.md"]
```

## Setup
This package requires Julia v1.5 or later. Refer the [official Julia documentation](https://julialang.org/downloads/) to install it for your system.

### Installation
To download this library, clone it from the julia package manager like so:
```julia
pkg> add https://github.com/intelligent-control-lab/ModelVerification.jl/
```

### Develop the package (for development)

**Deprecated once project is done and should be changed to "Building the package".**

Go to the package directory and start the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/). 
```julia
julia > ]
(@v1.9) > develop .
(@v1.9) > activate .
(@v1.9) > instantiate
```

This will enable development mode for the package. The dependency packages will also be installed. Some of the important ones are listed below. 
- [LazySets](https://juliareach.github.io/LazySets.jl/dev/)
- [JuMP](https://jump.dev/JuMP.jl/stable/)
- [Zygote](https://fluxml.ai/Zygote.jl/stable/)

## Overview of the package

## Tutorials

## Package Outline
```@contents
Pages = ["problem.md", "solvers.md"]
Depth = 3
```
# ModelVerification.jl

```@contents
Pages=["index.md"]
```

## Introduction
Deep Neural Network (DNN) is crucial in approximating nonlinear functions across diverse applications, such as computer vision and control. Verifying specific input-output properties can be a highly challenging task. To this end, we present [ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl), the only cutting-edge toolbox that contains a suite of state-of-the-art methods for verifying DNNs. This toolbox significantly extends and improves the previous version ([NeuralVerification.jl](https://sisl.github.io/NeuralVerification.jl/latest/)) and is designed to empower developers and machine learning practioners with robust tools for verifying and ensuring the trustworthiness of their DNN models.

### Key features:
- _Julia and Python integration_: Built on Julia programming language, [ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) leverages Julia's high-performance capabilities, ensuring efficient and scalable verification processes. Moreover, we provide the user with an easy, ready-to-use Python interface to exploit the full potential of the toolbox even without knowledge of the Julia language.
- _Different types of verification_: [ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) enables verification of several input-output specifications, such as reacability analysis, behavioral properties (e.g., to verify Deep Reinforcement Learning policies), or even robustness properties for Convolutional Neural Network (CNN). It also introduces new types of verification, not only for finding individual adversarial input, but for enumerating the entire set of unsafe zones for a given network and safety properties.
- _Verification benchmarks_: Compare our or your verification toolboxes against state-of-the-art benchmarks and evaluation criteria ([VNN-Comp 2023](https://vnncomp.christopher-brix.de/)). [ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) includes a collection of solvers and standard benchmarks to perform this evaluation efficiently.

## Setup
This toolbox requires Julia v1.5 or later. Refer the [official Julia documentation](https://julialang.org/downloads/) to install it for your system.

### Installation
To download this toolbox, clone it from the Julia package manager like so:

```Julia
pkg> add https://github.com/intelligent-control-lab/ModelVerification.jl/
```

### Develop the toolbox (for development)

_Deprecated once project is done and should be changed to "Building the package"._

Go to the toolbox directory and start the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/). 
```Julia
julia > ]
(@v1.9) > develop .
(@v1.9) > activate .
(@v1.9) > instantiate
```

This will enable development mode for the toolbox. The dependency packages will also be installed. Some of the important ones are listed below. 
- [LazySets](https://juliareach.github.io/LazySets.jl/dev/)
- [JuMP](https://jump.dev/JuMP.jl/stable/)
- [Zygote](https://fluxml.ai/Zygote.jl/stable/)

## Overview of the toolbox
![](./assets/overview.png)

[ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) receives input as a set consisting of:
- [Model](./network.md) to be verified,
- A [safety property](./problem.md) encoded as input-output specifications for the neural network,
- The [solver](./solvers.md) to be used for the formal verification process.

The toolbox's [output](./problem.md) varies depending on the type of verification we are performing. Nonetheless, at the end of the verification process, the response of the toolbox potentially allows us to obtain provable guarantees that a given safety property holds (or does not hold) for the model tested.

For more details on how the toolbox works, please refer to the [tutorial](#tutorials) below.

## Tutorials
- [MV-test](https://github.com/intelligent-control-lab/MV-test/blob/main/tutorial.ipynb)

## Toolbox Outline
![](./assets/MV_flow.png)

```@contents
Pages = ["problem.md", "network.md", "safety_spec.md", "branching.md", "propagate.md", "solvers.md"]
Depth = 3
```

## Python Interface
```@contents
Pages = ["nnet_converter.md"]
Depth = 3
```

## Benchmarks
```@contents
Pages = ["benchmark.md"]
Depth = 3
```
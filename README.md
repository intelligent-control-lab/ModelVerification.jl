# ModelVerification.jl

[![CI](https://github.com/intelligent-control-lab/ModelVerification.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/intelligent-control-lab/ModelVerification.jl/actions/workflows/CI.yml)
[![Doc](https://github.com/intelligent-control-lab/ModelVerification.jl/actions/workflows/documentation.yml/badge.svg)](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/)
[![codecov](https://codecov.io/github/intelligent-control-lab/ModelVerification.jl/graph/badge.svg?token=W0RYF98CVS)](https://codecov.io/github/intelligent-control-lab/ModelVerification.jl)
[![DOI](https://zenodo.org/badge/817002496.svg)](https://zenodo.org/badge/latestdoi/817002496)

## Introduction
Deep Neural Network (DNN) is crucial in approximating nonlinear functions across diverse applications, such as computer vision and control. Verifying specific input-output properties can be a highly challenging task. To this end, we present [ModelVerification.jl](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/#Introduction), the only cutting-edge toolbox that contains a suite of state-of-the-art methods for verifying DNNs. This toolbox significantly extends and improves the previous version ([NeuralVerification.jl](https://sisl.github.io/NeuralVerification.jl/latest/)) and is designed to empower developers and machine learning practioners with robust tools for verifying and ensuring the trustworthiness of their DNN models. Check out the [Documentation](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/) for more details. 

### Key features:
- _Julia and Python integration_: Built on Julia programming language, [ModelVerification.jl](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/#Key-features:) leverages Julia's high-performance capabilities, ensuring efficient and scalable verification processes. Moreover, we provide the user with an easy, ready-to-use Python interface to exploit the full potential of the toolbox even without knowledge of the Julia language (for future versions).
- _Different types of verification_: [ModelVerification.jl](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/#Key-features:) enables verification of several input-output specifications, such as reacability analysis, behavioral properties (e.g., to verify Deep Reinforcement Learning policies), or even robustness properties for Convolutional Neural Network (CNN). It also introduces new types of verification, not only for finding individual adversarial input, but for enumerating the entire set of unsafe zones for a given network and safety properties.
- _Visualization of intermediate verification results (reachable sets)_: [ModelVerification.jl](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/#Key-features:) enables the visualization of intermediate verification results in terms of reachable sets. In particular, our toolbox allows to plot the impactful features for the verification process and the correlated output reachable set (layer by layer) and thus to define new specific input-output specifications based on this information.
- _Verification benchmarks_: Compare our or your verification toolboxes against state-of-the-art benchmarks and evaluation criteria ([VNN-Comp 2023](https://vnncomp.christopher-brix.de/)). [ModelVerification.jl](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/#Key-features:) includes a collection of solvers and standard benchmarks to perform this evaluation efficiently.

## Run in a Docker
We have provided a pre-installed docker image based on `nvidia/cuda:12.2.0-devel-ubuntu22.04`. You can build the docker image using the following commands with the given docker file. It is tested on Linux OS with NVIDIA RTX A6000 GPU. Open an issue if you find difficulty with other platforms.

Build the Docker image:  
```bash
1. docker rmi modelverification-benchmark:latest  # # remove the old image
2. docker builder prune --force # clear the build cache
3. docker system prune -af  # remove all unused images, networks, and caches
4. cd ModelVerification.jl
5. docker build -t modelverification-benchmark . 
```

**(Optional but recommended)**  step 4. and 5. could be replaced with pulling pre-built docker image from docker hub:
```bash
docker pull huhanjiang666/modelverification-benchmark:latest
```

Run the container with GPU access:  
```bash
docker run --rm --gpus all -it modelverification-benchmark /bin/bash
```
Inside the container, run the ACASXU example:  
```bash
julia ModelVerification/vnncomp_scripts/test_ACASXU_single.jl
```
> **Note:** It’s **safe to ignore any CUDA warnings or errors**—the code will fall back to CPU without affecting correctness (only performance). 

Expected result:  
```
result = (value = "holds", time = 53.777557623, bytes = 13106889524, gctime = 4.549740117, gcstats = Base.GC_Diff(13106889524, 134300, 76, 132823415, 621545, 130236, 4549740117, 148, 0))
```

To run the full ACASXU property 1 benchmark as in the paper:
```bash
julia ModelVerification/vnncomp_scripts/test_ACASXU_all.jl
```


## Setup
This toolbox requires Julia v1.5 to v1.9.4. Any pull requests to support a higher version of Julia are welcomed. Refer the [official Julia documentation](https://julialang.org/downloads/) to install it for your system.

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
- [Flux](https://fluxml.ai/Flux.jl/stable/)
- [LazySets](https://juliareach.github.io/LazySets.jl/dev/)
- [JuMP](https://jump.dev/JuMP.jl/stable/)
- [Zygote](https://fluxml.ai/Zygote.jl/stable/)

## Overview of the toolbox
![](./assets/overview.png)

[ModelVerification.jl](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/index.html#Overview-of-the-toolbox) receives input as a set consisting of:
- [Model](./network.md) to be verified,
- A [safety property](./safety_spec.md) encoded as input-output specifications for the neural network,
- The [solver](./solvers.md) to be used for the formal verification process.

The toolbox's [output](./problem.md) varies depending on the type of verification we are performing. Nonetheless, at the end of the verification process, the response of the toolbox potentially allows us to obtain provable guarantees that a given safety property holds (or does not hold) for the model tested.

For more details on how the toolbox works, please refer to the [tutorial](#tutorials) below.

## Quickstart
Here is a simple example for verifying that the user-given safety property holds for a small deep neural network (DNN) with a single input node, two hidden layers with two ReLU nodes, and a single output node. We use the formal verification results obtained through the reachability analysis to get a provable answer whether the safety property holds.

First, we load the relevant libraries and the [ModelVerification.jl](https://intelligent-control-lab.github.io/ModelVerification.jl/dev/index.html#Quickstart) toolbox.
```Julia
using ModelVerification
using Flux
using LazySets
```

First, load the model.
```Julia
onnx_path = "models/small_nnet.onnx"
toy_model = ModelVerification.build_flux_model(onnx_path)
```

Suppose we want to verify that all inputs in $\mathcal{X}=[-2.5, 2.5]$ are mapped into $\mathcal{Y}=[18.5, 114.5]$. We encode this safety property using convex sets, provided by [LazySets](https://juliareach.github.io/LazySets.jl/dev/). 
```Julia
X = Hyperrectangle(low = [-2.5], high = [2.5]) # expected out: [18.5, 114.5]
Y = Hyperrectangle(low = [18.5], high = [114.5]) # here we expect the property holds
```

Now, we construct a _Problem_ instance. Note that [ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) converts the `.onnx` model into a [Flux](https://fluxml.ai/Flux.jl/stable/) model.
```Julia
problem = Problem(toy_model, X, Y)
```

Instantiate the `solver`, which in this case is [CROWN](https://arxiv.org/abs/1811.00866). We also need `search`, `split`, and `propagation` methods in addition to the `solver` and `Problem`.
```Julia
search_method = BFS(max_iter=100, batch_size=1)
split_method = Bisect(1)

use_gpu = false
lower_bound = true
upper_bound = true
solver = Crown(use_gpu, lower_bound, upper_bound)
```

Finally, we can verify that the safety property holds for this simple example!
```Julia
result = verify(search_method, split_method, solver, problem)
println(result)
println(result.status)
```

CROWN verifies that the input-output relationship holds!

## Tutorial Examples
- [Tutorials](https://github.com/intelligent-control-lab/ModelVerification.jl/blob/master/tutorial/tutorial.ipynb)
    - Example 1: Verifying a toy DNN with reachability analysis
    - Example 2: Verifying a CNN for robustness safety property
    - Example 3: Verifying a Deep Reinforcement Learning (DRL) policy for collision avoidance safety property
    - Example 4: Verifying Neural Control Barrier Function

## Toolbox Outline
![](./assets/overview_mvflow.png)
For detailed examples on how to use different functionalities provided by the toolbox, please refer to the [Tutorials](#tutorials). The pages below will direct you to the respective documentation for each category.

```@contents
Pages = ["toolbox_flow.md", "problem.md", "network.md", "safety_spec.md", "branching.md", "propagate.md", "solvers.md", "attack.md", "utils.md"]
Depth = 3
```

## Python Interface
_[Python Interface](./python_interface.md) is currently in development._

[ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) provides an interface with Python so that users who are not familiar with Julia can still use the toolbox via Python. Moreover, it provides [converters in Python](./nnet_converter.md) for converting between different neural network file formats, such as `.onnx`, `.pb`, `.pt`, `.h5`, and `.nnet`.

```@contents
Pages = ["nnet_converter.md", "python_interface.md"]
Depth = 3
```

## Citation

Please consider citing this toolbox if it is useful for your research.
```bibtex
@misc{wei2024MV,
  title = {ModelVerification.jl},
  author = {Wei, Tianhao and Hu, Hanjiang and Niu, Peizhi and Marzari, Luca and Yun, Kai S. and Luo, Xusheng and Liu, Changliu},
  howpublished = {\url{https://github.com/intelligent-control-lab/ModelVerification.jl}},
  year = {2024}
}
@article{wei2024modelverification,
  title={Modelverification. jl: a comprehensive toolbox for formally verifying deep neural networks},
  author={Wei, Tianhao and Marzari, Luca and Yun, Kai S and Hu, Hanjiang and Niu, Peizhi and Luo, Xusheng and Liu, Changliu},
  journal={arXiv preprint arXiv:2407.01639},
  year={2024}
}
```

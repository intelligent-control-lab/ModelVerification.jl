```@meta
CurrentModule = ModelVerification
```

```@contents 
Pages=["toolbox_flow.md"]
```

# Flow
![](./assets/overview_mvflow.png)

_This page serves to explain the overall flow of the toolbox. For examples and explanation on how to use specific methods (such as different [solvers](./solvers.md)), please refer to the [tutorials](./index.md#tutorials)._ 

In general, verification algorithms follow the paradigm of _Branch and Bound_. This process can be summarized into three steps:

1. split the input set into smaller sets, which we call "branches",
2. propagate the bound through the model for a given branch,
3. check whether the bound of the final layer satisfies the output specificaiton.

Repeat or terminate the process based on the result.

[ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) uses a modularized code structure to support various combinations of search methods, split methods, and solvers for a variety of geometric representations for the safety specifications and neural network architectures. After reading through this section, the user should have an overall idea of the flow of the toolbox and the design philosophy behind it.

### Definition of Terms
- **Instance**: combination of all the necessary information to define an "instance" of neural network verification problem. This is consisted of: 
    - [problem](./problem.md)
    - [solver](./solvers.md)
    - [search methods](./branching.md#search)
    - [split methods](./branching.md#split)
- **Propagation Method**: this is the bound propagation method used for verifying the problem. In other words, it is the choice of term to represent the ["solver"](./solvers.md): all the solvers in [ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) are represented as a propagation method. However, this is different from the [methods in `propagate.jl`](./propagate.md). This will be clearer in the following explanations.
- **Model / (Deep) Neural Network / Network**: these terms are used interchangeably and represent the deep neural network (DNN) to be verified.

## 1. Creating an instance: _what kind of verification problem do you want to solve?_
Let's first create an instance. An instance contains all the information required to run the [`verify`](@ref) function. This function does the heavy-lifting where the verification problem is solved. As long as the user properly defines the problem and solver methods, this is the only function the user has to call. To run [`verify`](@ref), the user has to provide the following arguments. These collectively defines an "instance":

- [`SearchMethod`](@ref): Algorithm for iterating through the branches, such as `BFS` (breadth-first search) and `DFS` (depth-first search).
- [`SplitMethod`](@ref): Algorithm for splitting an unknown branch into smaller pieces for further refinement. This is also used in the first step of [`verify`](@ref) to populate the branches bank. In other words, it splits the input specification into branches to facilitate the propagation process.
- [`PropMethod`](@ref): Solver used to verify the problem, such as `Ai2` and `Crown`.
- [`Problem`](@ref): Problem to be verified. It is consisted of a [`Network`](@ref), and [input and output specifications](./safety_spec.md).




## 2. Verifying the instance: _spinning through the branches - where the magic happens!_


```@docs
verify
```


## 3. Results and how to interpret them: _so is my model good to go?_
The result is either a `BasicResult`, `CounterExampleResult`, 
`AdversarialResult`, `ReachabilityResult`, `EnumerationResult`, or timeout. 
For each `Result`, the `status` field is either `:violated`, `:verified`, 
`:unknown`, or `:timeout`.
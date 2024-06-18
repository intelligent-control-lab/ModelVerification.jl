"""
    ODETaylor <: SequentialForwardProp

`ODETaylor` performs reachability analysis for Neural ODE with Taylor method.
It works for networks with smooth activations.

## Problem Requirement
1. Network: any depth, smooth activation (e.g. tanh or sigmoid)
2. Input: Array of AbstractPolytope, i.e., union of polytopes.
3. Output: Array of AbstractPolytope, i.e., union of polytopes.

## Returns
`BasicResult(:holds)`, `BasicResult(:violated)`

## Method
Reachability analysis using split and join.

## Property
Sound and complete.

"""
struct ODETaylor <: ODEProp 
    t_span::FloatType[]
end
ODETaylor(;t_span=0.0) = ODETaylor(t_span)

"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, 
                    prop_method::ODETaylor, problem::Problem)

Preprocessing of the `Problem` to be solved. This method converts the model to a 
bounded computational graph, makes the input specification compatible with the 
solver, and returns the model information and preprocessed `Problem`. This in 
turn also initializes the branch bank.

## Arguments
- `search_method` (`SearchMethod`): Method to search the branches.
- `split_method` (`SplitMethod`): Method to split the branches.
- `prop_method` (`ODETaylor`): Solver to be used, specifically the 
    `ODETaylor`.
- `problem` (`Problem`): Problem to be preprocessed to better fit the solver.

## Returns
- `model_info`, a structure containing the information of the neural network to 
    be verified.
- `Problem` after processing the initial input specification and model.
"""
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ODETaylor, problem::Problem)
    model_info = problem.Flux_model
    return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
end

"""
    prepare_method(prop_method::ODEProp, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info)

Initialize the bound for Neural ODEs.

## Agruments
- `prop_method` (`ODEProp`): Propagation method, i.e., the solver.
- `batch_input` (`AbstractVector`): Batch of inputs.
- `batch_output` (`AbstractVector`): Batch of outputs.
- `batch_inheritance` (`AbstractVector`): Batch of inheritance, can be used to inheritate pre-act-bound from the parent branch
- `model_info`: Structure containing the information of the neural network to
    be verified.

## Returns
- `batch_output`: Batch of outputs.
- `batch_info`: Dictionary containing information of each node in the model.
"""
function prepare_method(prop_method::ODEProp, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info)
    batch_info = Dict(:batch_input => batch_input)
    return batch_output, batch_info
end

"""
    check_inclusion(prop_method::ODETaylor, model, input::LazySet, 
    reach::TaylorModelReachSet, output::LazySet)

Determines whether the reachable set, `reach`, is within the valid 
output specified by a `LazySet`. This function achieves this by directly 
checking if the reachable set `reach` is a subset of the set of valid outputs 
`output`. If not, it attempts to find a counterexample and returns the 
appropriate `Result`.

## Arguments
- `prop_method` (`ODETaylor`): Solver being used.
- `model`: Neural network model that is to be verified.
- `input` (`LazySet`): Input specification supported by `LazySet`.
- `reach` (`TaylorModelReachSet`): Reachable set resulting from the propagation of 
    `input` through the `model`.
- `output` (`LazySet`) : Set of valid outputs represented with a `LazySet`.

## Returns
- `ReachabilityResult(:holds, [reach])` if `reach` is a subset of `output`.
- `CounterExampleResult(:unknown)` if `reach` is not a subset of `output`, but 
    cannot find a counterexample.
- `CounterExampleResult(:violated, x)` if `reach` is not a subset of `output`, 
    and there is a counterexample.
"""
function check_inclusion(prop_method::ODETaylor, model, input::LazySet, 
                         reach::TaylorModelReachSet, output::LazySet)
    # println(reach)
    # println(⊆(reach, output))
    ⊆(reach, output) && return ReachabilityResult(:holds, [reach])
    x = LazySets.center(input)
    ∈(model(x), output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end
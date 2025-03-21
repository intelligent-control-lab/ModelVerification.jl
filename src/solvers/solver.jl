"""
    ForwardProp <: PropMethod

Abstract type representing solvers that use forward propagation.
"""
abstract type ForwardProp <: PropMethod end

"""
    BackwardProp <: PropMethod

Abstract type representing solvers that use backward propagation.
"""
abstract type BackwardProp <: PropMethod end

"""
    ODEProp <: PropMethod

Abstract type representing solvers that use backward propagation.
"""
abstract type ODEProp <: ForwardProp end


"""
    SequentialForwardProp <: ForwardProp
"""
abstract type SequentialForwardProp <: ForwardProp end

"""
    SequentialBackwardProp <: ForwardProp
"""
abstract type SequentialBackwardProp <: BackwardProp end

"""
    BatchForwardProp <: ForwardProp
"""
abstract type BatchForwardProp <: ForwardProp end

"""
    BatchBackwardProp <: BackwardProp
"""
abstract type BatchBackwardProp <: BackwardProp end

"""
    Bound

Abstract type representing bounds.
"""
abstract type Bound end

"""
    init_batch_bound(prop_method::ForwardProp, batch_input, batch_output)

Returns a list of the input specifications (geometries) for the given batch of 
inputs. This is for `ForwardProp` solvers. Each input specification is 
processed to fit the geometric representation used by the solver.

## Arguments
- `prop_method` (`ForwardProp`): Solver that uses forward propagation method.
- `batch_input`: Array of inputs.
- `batch_output`: Array of outputs.

## Returns
- List of the input specifications for the given batch of inputs.
"""
function init_batch_bound(prop_method::ForwardProp, batch_input, batch_output)
    return [init_bound(prop_method, input) for input in batch_input]
end

"""
    init_batch_bound(prop_method::BackwardProp, batch_input, batch_output)

Returns a list of the output specifications (geometries) for the given batch of 
outputs. This is for `BackwardProp` solvers. Each input specification is 
processed to fit the geometric representation used by the solver.

## Arguments
- `prop_method` (`BackwardProp`): Solver that uses backward propagation method.
- `batch_input`: Array of inputs.
- `batch_output`: Array of outputs.

## Returns
- List of the output specifications for the given batch of outputs.
"""
function init_batch_bound(prop_method::BackwardProp, batch_input, batch_output)
    return [init_bound(prop_method, output) for output in batch_output]
end

"""
    init_bound(prop_method::ForwardProp, input)

Returns the geometry representation used to encode the input specification.
This is for `ForwardProp` solvers. 

## Arguments
- `prop_method` (`ForwardProp`): Solver that uses forward propagation method. 
- `input`: Geometry representation used to encode the input specification.

## Returns
- `input`: Geometry representation used to encode the input specification.
"""
function init_bound(prop_method::ForwardProp, input)
    return input
end

"""
    init_bound(prop_method::BackwardProp, output)

Returns the geometry representation used to encode the output specification. 
This is for `BackwardProp` solvers.

## Arguments
- `prop_method` (`BackwardProp`): Solver that uses backward propagation method. 
- `output`: Geometry representation used to encode the output specification.

## Returns
- `output`: Geometry representation used to encode the output specification.
"""
function init_bound(prop_method::BackwardProp, output)
    return output
end

"""
    process_bound(prop_method::PropMethod, batch_bound, batch_out_spec, model_info, batch_info)

Returns the list of bounds resulting from the propagation and the information of
the batch.

## Arguments
- `prop_method` (`PropMethod`): Solver.
- `batch_bound`: List of the bounds for the given batch.
- `batch_out_spec`: List of the output specifications for the given batch of 
    outputs.
- `model_info`: Structure containing the information of the neural network to be 
    verified.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `batch_bound`: List of the bounds for the given batch.
- `batch_info`: Dictionary containing information of each node in the model.
"""
function process_bound(prop_method::PropMethod, batch_bound, batch_out_spec, model_info, batch_info)
    return batch_bound, batch_info
end

"""
    init_propagation(prop_method::ForwardProp, batch_input, batch_output, model_info)

Returns a dictionary containing the information of each node in the model. This 
function is for `ForwardProp` solvers, and is mainly concerned with initializing 
the dictionary, `batch_info`, and populating it with the initial bounds for the 
starting node. For the starting node of the model, the `:bound` key is mapped 
to the list of input specifications.

## Arguments
- `prop_method` (`ForwardProp`): Solver that uses forward propagation.
- `batch_input`: List of inputs.
- `batch_output`: List of outputs.
- `model_info`: Structure containing the information of the neural network to 
    be verified.

## Returns
- `batch_info`: Dictionary containing information of each node in the model.
"""
function init_propagation(prop_method::ForwardProp, batch_input, batch_output, model_info)
    @assert length(model_info.start_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.start_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input, batch_output)
    return batch_info
end

"""
    init_propagation(prop_method::BackwardProp, batch_input, batch_output, model_info)

Returns a dictionary containing the information of each node in the model. This 
function is for `BackwardProp` solvers, and is mainly concerned with 
initializing the dictionary, `batch_info`, and populating it with the initial 
bounds for the starting node. For the starting node of the model, the `:bound` 
key is mapped to the list of input specifications.

## Arguments
- `prop_method` (`BackwardProp`): Solver that uses backward propagation.
- `batch_input`: List of inputs.
- `batch_output`: List of outputs.
- `model_info`: Structure containing the information of the neural network to 
    be verified.

## Returns
- `batch_info`: Dictionary containing information of each node in the model.
"""
function init_propagation(prop_method::BackwardProp, batch_input, batch_output, model_info)
    # @show model_info.final_nodes
    @assert length(model_info.final_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.final_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input, batch_output)
    return batch_info
end
 
"""
    prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info)

Initialize the bound of the start node of the computational graph based on the 
solver (`prop_method`).

## Agruments
- `prop_method` (`PropMethod`): Propagation method, i.e., the solver.
- `batch_input` (`AbstractVector`): Batch of inputs.
- `batch_output` (`AbstractVector`): Batch of outputs.
- `batch_inheritance` (`AbstractVector`): Batch of inheritance, can be used to inheritate pre-act-bound from the parent branch
- `model_info`: Structure containing the information of the neural network to
    be verified.

## Returns
- `batch_output`: Batch of outputs.
- `batch_info`: Dictionary containing information of each node in the model.
"""
function prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, batch_output, model_info)
    return batch_output, batch_info
end

"""
    check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, batch_reach::AbstractArray, batch_output::AbstractArray)

Determines whether the reachable sets, `batch_reach`, are within the respective 
valid output sets, `batch_output`.

## Arguments
- `prop_method` (`ForwardProp`): Solver being used.
- `model`: Neural network model that is to be verified.
- `input` (`AbstractArray`): List of input specifications.
- `reach` (`AbstractArray`): List of reachable sets.
- `output` (`AbstractArray`) : List of sets of valid outputs.

## Returns
List of a combination of the following components:

- `ReachabilityResult(:holds, [reach])` if `reach` is a subset of `output`.
- `CounterExampleResult(:unknown)` if `reach` is not a subset of `output`, but 
    cannot find a counterexample.
- `CounterExampleResult(:violated, x)` if `reach` is not a subset of `output`, 
    and there is a counterexample.
"""
function check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, batch_reach::AbstractArray, batch_output::AbstractArray)
    results = [check_inclusion(prop_method, model, batch_input[i], batch_reach[i], batch_output[i]) for i in eachindex(batch_reach)]
    return results
end


"""
    custom_check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, 
                           batch_reach::AbstractArray, batch_output::AbstractArray, check_inclusion_helper)

Custom method to determine whether the reachable sets (`batch_reach`) are within the respective 
valid output sets (`batch_output`), using a helper function for finer-grained checks.

## Arguments

- `prop_method` (`ForwardProp`): Solver being used for forward propagation.
- `model`: Neural network model that is to be verified.
- `batch_input` (`AbstractArray`): List of input specifications.
- `batch_reach` (`AbstractArray`): List of reachable sets.
- `batch_output` (`AbstractArray`): List of sets of valid outputs.
- `check_inclusion_helper` (Function): Helper function to evaluate whether a specific reachable 
  set is within the corresponding valid output set.

## Returns

A list of results with the following components:

- `ReachabilityResult(:holds, reach)`: Indicates that the reachable set `reach` is fully within the 
  corresponding output set.
- `CounterExampleResult(:unknown)`: Indicates that the reachable set is not fully within the output 
  set, but no counterexample can be found.
- `CounterExampleResult(:violated, x)`: Indicates that the reachable set is not within the output 
  set, and a counterexample input `x` was found that violates the output constraint.

## Method Description

This function iterates over the batched reachable and output sets. For each pair, it checks 
inclusion using `check_inclusion_helper`. If the reachable set is a subset of the output set, it 
records the result as `:holds`. Otherwise, it attempts to refine the analysis by testing the center 
of the input zonotope. If further refinement is inconclusive, it records the result as `:unknown`. 
If a counterexample is found, it records the result as `:violated`.
"""

function custom_check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, batch_reach::AbstractArray, batch_output::AbstractArray, check_inclusion_helper::Function)
    batch_result = []
    for i in eachindex(batch_reach)
        # Check inclusion using the helper method
        result = check_inclusion_helper(
            prop_method, 
            model,
            batch_input[i], 
            batch_reach[i], 
            batch_output[i]
        )
        if result
            # If inclusion holds, add to results
            push!(batch_result, ReachabilityResult(:holds, batch_reach[i]))
        else
            # Prepare counterexample or violation analysis
            x = batch_input[i].center
            # Compute the output and create a zonotope
            y = reshape(model(x), :)  # Flatten output
            generators = Float64.(zeros(size(y)))  # Initialize zero generators
            center_reach = Zonotope(y, reshape(generators, :, 1))  # Create a zonotope
    
            # Recheck inclusion for the refined input
            result = check_inclusion_helper(
                prop_method, 
                model, 
                x, 
                center_reach, 
                batch_output[i]
            )
            
            if result
                # If still uncertain, add an unknown result
                push!(batch_result, CounterExampleResult(:unknown))
            else
                # If violated, record the counterexample
                push!(batch_result, CounterExampleResult(:violated, x))
            end
        end
    end
    return batch_result
end

"""
    get_inheritance(prop_method::PropMethod, batch_info::Dict, batch_idx::Int)

Extract useful informations from batch_info.
These information will later be inheritated by the new branch created by split.

## Arguments
- `prop_method` (`ForwardProp`): Solver being used.
- `batch_info` (`Dict`): all the information collected in propagation.
- `batch_idx`: the index of the interested branch in the batch.
- `model_info`: the general computational graph

## Returns
- `inheritance`: a dict that contains all the information will be inheritated.
"""
function get_inheritance(prop_method::PropMethod, batch_info::Dict, batch_idx::Int, model_info)
    return nothing
end
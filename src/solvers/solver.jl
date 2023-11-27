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
    @assert length(model_info.final_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.final_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input, batch_output)
    return batch_info
end
 
"""
    prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, model_info)

Initialize the bound of the start node of the computational graph based on the 
solver (`prop_method`).

## Agruments
- `prop_method` (`PropMethod`): Propagation method, i.e., the solver.
- `batch_input` (`AbstractVector`): Batch of inputs.
- `batch_output` (`AbstractVector`): Batch of outputs.
- `model_info`: Structure containing the information of the neural network to
    be verified.

## Returns
- `batch_output`: Batch of outputs.
- `batch_info`: Dictionary containing information of each node in the model.
"""
function prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
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

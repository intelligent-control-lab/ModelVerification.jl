"""
    ForwardProp <: PropMethod
"""
abstract type ForwardProp <: PropMethod end

"""
    BackwardProp <: PropMethod
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
"""
function init_batch_bound(prop_method::ForwardProp, batch_input, batch_output)
    return [init_bound(prop_method, input) for input in batch_input]
end

"""
    init_batch_bound(prop_method::BackwardProp, batch_input, batch_output)
"""
function init_batch_bound(prop_method::BackwardProp, batch_input, batch_output)
    return [init_bound(prop_method, output) for output in batch_output]
end

"""
    init_bound(prop_method::ForwardProp, input)
"""
function init_bound(prop_method::ForwardProp, input)
    return input
end

"""
    init_bound(prop_method::BackwardProp, output)
"""
function init_bound(prop_method::BackwardProp, output)
    return output
end

"""
    process_bound(prop_method::PropMethod, batch_bound, batch_out_spec, model_info, batch_info)
"""
function process_bound(prop_method::PropMethod, batch_bound, batch_out_spec, model_info, batch_info)
    return batch_bound, batch_info
end

"""
    init_propagation(prop_method::ForwardProp, batch_input, batch_output, model_info)


"""
function init_propagation(prop_method::ForwardProp, batch_input, batch_output, model_info)
    @assert length(model_info.start_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.start_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input, batch_output)
    return batch_info
end

"""
    init_propagation(prop_method::BackwardProp, batch_input, batch_output, model_info)
"""
function init_propagation(prop_method::BackwardProp, batch_input, batch_output, model_info)
    @assert length(model_info.final_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.final_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input, batch_output)
    return batch_info
end
 
"""
    prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
"""
function prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, batch_output, model_info)
    return batch_output, batch_info
end

"""
    check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, batch_reach::AbstractArray, batch_output::AbstractArray)
"""
function check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, batch_reach::AbstractArray, batch_output::AbstractArray)
    results = [check_inclusion(prop_method, model, batch_input[i], batch_reach[i], batch_output[i]) for i in eachindex(batch_reach)]
    return results
end

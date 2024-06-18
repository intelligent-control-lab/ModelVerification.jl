"""
    propagate_layer(prop_method, layer::typeof(flatten), 
                     bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a flatten layer. I.e., it flattens 
the `ImageStarBound` into a `Star` type.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(flatten)`): The layer operation to be used for propagation.
- `bound` (`ImageStarBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The flattened bound of the output layer represented in `Star` type.
"""
propagate_layer(prop_method, layer::typeof(flatten), bound::ImageStarBound, batch_info) = 
    Star(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)), HPolyhedron(bound.A, bound.b))

"""
    propagate_layer(prop_method, layer::typeof(flatten), 
                     bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a flatten layer. I.e., it flattens 
the `ImageZonoBound` into a `Zonotope` type.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(flatten)`): The layer operation to be used for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The flattened bound of the output layer represented in `Zonotope` type.
"""
propagate_layer(prop_method, layer::typeof(flatten), bound::ImageZonoBound, batch_info) =
    Zonotope(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)))



function propagate_layer_batch(prop_method::IBP, layer::typeof(Flux.flatten), bound::IBPBound, batch_info)
    sz = size(bound.batch_low)
    flattened_dim = prod(sz[1:end-1])
    new_low = reshape(bound.batch_low, (flattened_dim, sz[end]))
    new_up = reshape(bound.batch_up, (flattened_dim, sz[end]))
    return IBPBound(new_low, new_up)
end


function propagate_layer_batch(prop_method::Crown, layer::typeof(Flux.flatten), bound::CrownBound, batch_info)
    bound, _ = convert_CROWN_Bound_batch(bound)
    return bound
end


function propagate_layer_batch(prop_method::BetaCrown, layer::typeof(Flux.flatten), bound::BetaCrownBound, batch_info)

    node = batch_info[:current_node]
    @assert !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
    
    # @show node
    size_before_layer = batch_info[node][:size_before_layer][1:3]
    # @show batch_info[node][:size_before_layer], batch_info[node][:size_after_layer]
    size_after_layer = [batch_info[node][:size_after_layer][1]]

    lA_W = uA_W = nothing 
    lA_x = prop_method.bound_lower ? flatten_bound_oneside(size_before_layer, size_after_layer, batch_info[:batch_size]) : nothing
    uA_x = prop_method.bound_upper ? flatten_bound_oneside(size_before_layer,size_after_layer, batch_info[:batch_size]) : nothing
    New_bound = BetaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return New_bound
end

"""
flatten_bound_oneside(kernel_size, stride, pad, batch_data_min, batch_data_max,size_after_layer, batch_size)

"""
function flatten_bound_oneside(size_before_layer, size_after_layer, batch_size)
    function bound_flatten(x)
        x_weight = x[1]
        @assert size(x_weight)[2] == size_after_layer[1]
        x_bias = zeros(size(x[2]))
        batch_reach = x_weight
        return [batch_reach, x_bias]
    end
    return bound_flatten
    # push!(last_A, find_w_b)
    # return last_A
end
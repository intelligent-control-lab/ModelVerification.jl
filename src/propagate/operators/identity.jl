"""
    propagate_layer(prop_method, σ::typeof(identity), bound, batch_info)

Propagate the bounds through the identity activation layer.

## Arguments
- `prop_method`: Propagation method.
- `σ`: Identity activation function.
- `bound`: Bounds of the input.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `bound`: Bounds of the output, which is equivalent to the bounds of the input.
"""
function propagate_layer(prop_method::ForwardProp, σ::typeof(identity), bound, batch_info)
    return bound
end
function propagate_layer(prop_method::BackwardProp, σ::typeof(identity), bound, batch_info)
    return bound
end
function propagate_layer_batch(prop_method::BackwardProp, σ::typeof(identity), bound, batch_info)
    return bound
end
function propagate_layer_batch(prop_method::ForwardProp, σ::typeof(identity), bound, batch_info)
    return bound
end
function propagate_layer_batch(prop_method::ForwardProp, σ::typeof(identity), bound::AbstractArray, batch_info)
    return bound
end

function propagate_layer_batch(prop_method::BetaCrown, σ::typeof(identity), bound::BetaCrownBound, batch_info)
    return BetaCrownBound(identity, identity, nothing, nothing, bound.batch_data_min, bound.batch_data_max, bound.img_size)
end
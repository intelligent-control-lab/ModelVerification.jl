
function forward_linear(prop_method::ForwardProp, layer::Dense, reach::LazySet, info)
    reach, info = affine_map(layer, reach), info
    return reach, info
end

# Ai2 Box
function forward_linear(prop_method::Box, layer::Dense, reach::LazySet, info)
    isa(reach, AbstractPolytope) || throw("Ai2 only support AbstractPolytope type branches.")
    reach, info = approximate_affine_map(layer, reach), info
    return reach, info
end  

function batch_interval_map(W::AbstractMatrix{N}, l::AbstractArray, u::AbstractArray) where N
    pos_W = max.(W, zero(N))
    neg_W = min.(W, zero(N))
    l_new = batched_mul(pos_W, l) + batched_mul(neg_W, u) # reach_dim x input_dim+1 x batch
    u_new = batched_mul(pos_W, u) + batched_mul(neg_W, l) # reach_dim x input_dim+1 x batch
    return (l_new, u_new)
end

function forward_linear_batch(prop_method::Crown, layer::Dense, bound::CrownBound, batch_info)
    # out_dim x in_dim * in_dim x X_dim x batch_size
    output_Low, output_Up = batch_interval_map(layer.weight, bound.batch_Low, bound.batch_Up)
    @assert !any(isnan, output_Low) "contains NaN"
    @assert !any(isnan, output_Up) "contains NaN"
    output_Low[:, end, :] .+= layer.bias
    output_Up[:, end, :] .+= layer.bias
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max)
    return new_bound, batch_info
end

function forward_linear(prop_method::AlphaCrown, layer::Dense, bound::CrownBound, batch_info)
    # out_dim x in_dim * in_dim x X_dim x batch_size
    output_Low, output_Up = batch_interval_map(layer.weight, bound.batch_Low, bound.batch_Up)
    @assert !any(isnan, output_Low) "contains NaN"
    @assert !any(isnan, output_Up) "contains NaN"
    output_Low[:, end, :] .+= layer.bias
    output_Up[:, end, :] .+= layer.bias
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max)
    # l, u = compute_bound(new_bound)
    return new_bound, batch_info
end

# Ai2z, Ai2h
function forward_linear(prop_method::ForwardProp, layer::Dense, batch_reach::AbstractArray, batch_info)
    all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
    batch_reach = identity.(batch_reach) # identity. converts Vector{Any} to Vector{AbstractPolytope}
    batch_reach, batch_info = affine_map(layer, batch_reach), batch_info
    return batch_reach, batch_info
end

# Ai2 Box
function forward_linear(prop_method::Box, layer::Dense, batch_reach::AbstractArray, batch_info)
    all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
    batch_reach = identity.(batch_reach) # identity. converts Vector{Any} to Vector{AbstractPolytope}
    batch_reach, batch_info = approximate_affine_map(layer, batch_reach), batch_info
    return batch_reach, batch_info
end  

# function forward_linear(prop_method::Neurify, layer::Dense, batch_reach::LinearBound, batch_info)
#     output_Low, output_Up = batch_interval_map(layer.weights, batch_reach.Low, batch_reach.Up)
#     output_Low[:, end, :] += layer.bias
#     output_Up[:, end, :] += layer.bias
#     output_batch_reach = LinearBound(output_Low, output_Up, batch_reach.domain)
#     return output_batch_reach, batch_info
# end
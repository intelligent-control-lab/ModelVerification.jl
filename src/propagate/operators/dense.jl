
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
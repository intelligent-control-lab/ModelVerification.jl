
function propagate_linear(prop_method::ForwardProp, layer::Dense, reach::LazySet, batch_info)
    reach = affine_map(layer, reach)
    return reach
end

# Ai2 Box
function propagate_linear(prop_method::Box, layer::Dense, reach::LazySet, batch_info)
    isa(reach, AbstractPolytope) || throw("Ai2 only support AbstractPolytope type branches.")
    reach = approximate_affine_map(layer, reach)
    return reach
end  

function batch_interval_map(W::AbstractMatrix{N}, l::AbstractArray, u::AbstractArray) where N
    pos_W = max.(W, zero(N))
    neg_W = min.(W, zero(N))
    l_new = batched_mul(pos_W, l) + batched_mul(neg_W, u) # reach_dim x input_dim+1 x batch
    u_new = batched_mul(pos_W, u) + batched_mul(neg_W, l) # reach_dim x input_dim+1 x batch
    return (l_new, u_new)
end

function propagate_linear_batch(prop_method::Crown, layer::Dense, bound::CrownBound, batch_info)
    # out_dim x in_dim * in_dim x X_dim x batch_size
    output_Low, output_Up = batch_interval_map(layer.weight, bound.batch_Low, bound.batch_Up)
    @assert !any(isnan, output_Low) "contains NaN"
    @assert !any(isnan, output_Up) "contains NaN"
    output_Low[:, end, :] .+= layer.bias
    output_Up[:, end, :] .+= layer.bias
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max)
    return new_bound
end

function propagate_linear(prop_method::AlphaCrown, layer::Dense, bound::CrownBound, batch_info)
    # out_dim x in_dim * in_dim x X_dim x batch_size
    output_Low, output_Up = batch_interval_map(layer.weight, bound.batch_Low, bound.batch_Up)
    @assert !any(isnan, output_Low) "contains NaN"
    @assert !any(isnan, output_Up) "contains NaN"
    output_Low[:, end, :] .+= layer.bias
    output_Up[:, end, :] .+= layer.bias
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max)
    # l, u = compute_bound(new_bound)
    return new_bound
end

# Ai2z, Ai2h
function propagate_linear(prop_method::ForwardProp, layer::Dense, batch_reach::AbstractArray, batch_info)
    all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
    batch_reach = identity.(batch_reach) # identity. converts Vector{Any} to Vector{AbstractPolytope}
    batch_reach = affine_map(layer, batch_reach)
    return batch_reach
end

# Ai2 Box
function propagate_linear(prop_method::Box, layer::Dense, batch_reach::AbstractArray, batch_info)
    all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
    batch_reach = identity.(batch_reach) # identity. converts Vector{Any} to Vector{AbstractPolytope}
    batch_reach = approximate_affine_map(layer, batch_reach)
    return batch_reach
end  

# function propagate_linear(prop_method::Neurify, layer::Dense, batch_reach::LinearBound)
#     output_Low, output_Up = batch_interval_map(layer.weights, batch_reach.Low, batch_reach.Up)
#     output_Low[:, end, :] += layer.bias
#     output_Up[:, end, :] += layer.bias
#     output_batch_reach = LinearBound(output_Low, output_Up, batch_reach.domain)
#     return output_batch_reach
# end

function _preprocess(node, batch_info, bias = nothing)
    if !isnothing(bias)
        if batch_info[node]["beta"] != 1.0 
            bias = batch_info[node]["beta"] .* bias
        end
    end
    return bias
end

function bound_oneside(last_A, weight, bias)
    if isnothing(last_A)
        return nothing, 0
    end

    weight = reshape(weight, (size(weight)..., 1)) 
    weight = repeat(weight, 1, 1, size(last_A)[end]) #add batch dim in weight
    weight = permutedims(weight, (2, 1, 3)) #permute the 1st and 2sd dims for batched_mul
    new_A = NNlib.batched_mul(weight, last_A) #note: must be weight * last_A, not last_A * weight
    
    if !isnothing(bias)
        bias = reshape(bias, (size(bias)..., 1, 1)) #add input dim in weight
        bias = repeat(bias, 1, 1, size(last_A)[end]) #add batch dim in weight
        bias = permutedims(bias, (2, 1, 3))
        sum_bias = NNlib.batched_mul(bias, last_A)
    else
        sum_bias = 0.0
    end

    return next_A, sum_bias
end

function propagate_linear_batch(prop_method::AlphaCrown, layer::Dense, node, bound::AlphaCrownBound, batch_info)
    last_lA = bound.lower_A_x
    last_uA = bound.upper_A_x

    #TO DO: we haven't consider the perturbation in weight and bias
    bias_lb = _preprocess(node, batch_info, layer.bias)
    bias_ub = _preprocess(node, batch_info, layer.bias)
    lA_y = uA_y = lA_bias = uA_bias = nothing
    lbias = ubias = 0
    batch_size = !isnothing(last_lA) ? size(last_lA)[end] : size(last_lA)[end]

    if !batch_info[node]["weight_ptb"] && (!batch_info[node]["bias_ptb"] || isnothing(layer.bias))
        weight = layer.weight
        bias = bias_lb
        
        #= index = last_lA
        coeffs = nothing
        
        if !isnothing(weight)
            new_weight = weight[index, :] #get the parameters that correspond to unstable neuron
            lA_x = reshape(new_weight, (size(new_weight)..., 1))
        end
        if !isnothing(bias)
            new_bias = bias[index, :] #get the parameters that correspond to unstable neuron
            lbias = reshape(new_bias, (size(new_bias)..., 1))
        end
        uA_x, ubias = lA_x, lbias =#
        
        lA_x, lbias = bound_oneside(last_lA, weight, bias)
        uA_x, ubias = bound_oneside(last_uA, weight, bias)

        bound = AlphaCrownBound(bound.batch_Low, bound.batch_Up, lA_x, uA_x, lA_y, uA_y, lbias, ubias)
        return bound
    end
end
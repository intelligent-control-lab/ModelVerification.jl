
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
        if batch_info[node][:beta] != 1.0 
            bias = batch_info[node][:beta] .* bias
        end
    end
    return bias
end

function dense_bound_oneside(last_A, weight, bias, batch_size)
    if isnothing(last_A)
        #return nothing, 0
        return nothing, nothing
    end
    weight = reshape(weight, (size(weight)..., 1)) 
    weight = repeat(weight, 1, 1, batch_size) #add batch dim in weight
    #New_A = NNlib.batched_mul(last_A, weight) 

    if !isnothing(bias)
        New_bias = []
        bias = reshape(bias, (size(bias)..., 1))
        bias = repeat(bias, 1, batch_size) 
        # println()
        #New_bias = NNlib.batched_mul(last_A, bias)
        push!(last_A, x -> [NNlib.batched_mul(x[1], weight), NNlib.batched_mul(x[1], bias) .+ x[2]]) 
    else
        push!(last_A, x -> [NNlib.batched_mul(x[1], weight), x[2]]) 
    end
    return last_A
end

function propagate_linear_batch(prop_method::AlphaCrown, layer::Dense, bound::AlphaCrownBound, batch_info)
    node = batch_info[:current_node]
    #TO DO: we haven't consider the perturbation in weight and bias
    bias_lb = _preprocess(node, batch_info, layer.bias)
    bias_ub = _preprocess(node, batch_info, layer.bias)
    lA_W = uA_W = lA_bias = uA_bias = nothing

    #lower_A = bound.lower_A_x
    #upper_A = bound.upper_A_x
    if !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
        weight = layer.weight
        bias = bias_lb
        if prop_method.bound_lower
            lA_x = dense_bound_oneside(bound.lower_A_x, weight, bias, batch_info[:batch_size])
        else
            lA_x = nothing
        end
        if prop_method.bound_upper
            uA_x = dense_bound_oneside(bound.upper_A_x, weight, bias, batch_info[:batch_size])
        else
            uA_x = nothing
        end
        New_bound = AlphaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max)
        return New_bound
    end
end
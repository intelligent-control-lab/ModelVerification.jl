using Plots

"""
    propagate_layer(prop_method::ForwardProp, layer::Dense, 
                     reach::LazySet, batch_info)

Propagate the bounds through the dense layer. It operates an affine 
transformation on the given input bound and returns the output bound.
                     
## Arguments
- `prop_method` (`ForwardProp`): Forward propagation method used for the 
    verification process. This is one of the solvers used to verify the given 
    model.                  
- `layer` (`Dense`): Dense layer of the model.
- `reach` (`LazySet`): Bound of the input.
- `batch_info`: Dictionary containing information of each node in the 
    model.

## Returns
- `reach` (`LazySet`): Bound of the output after affine transformation.
"""
function propagate_layer(prop_method::ForwardProp, layer::Dense, reach::LazySet, batch_info)
    reach = affine_map(layer, reach)
    # @show reach
    # display(plot(reach, title=typeof(prop_method), xlim=[-3,3], ylim=[-3,3]))
    return reach
end

"""
    propagate_layer(prop_method::ExactReach, layer::Dense, 
                     reach::ExactReachBound, batch_info)

Propagate the bounds through the dense layer. It operates an affine 
transformation on the given input bound and returns the output bound for 
`ExactReach` solver.

## Arguments
- `prop_method` (`ExactReach`): Exact reachability method used for the 
    verification process. This is one of the solvers used to verify the given 
    model.
- `layer` (`Dense`): Dense layer of the model.
- `reach` (`ExactReachBound`): Bound of the input, represented by 
    `ExactReachBound` type, which is a vector of `LazySet` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `reach` (`ExactReachBound`): Bound of the output after affine transformation, 
    which is represented by `ExactReachBound` type.
"""
function propagate_layer(prop_method::ExactReach, layer::Dense, reach::ExactReachBound, batch_info)
    bounds = []
    cnt = 0
    for bound in reach.polys
        # cnt += 1
        # println("cnt: ", cnt)
        # println(layer)
        # println(bound)
        # sleep(0.1)
        nb = affine_map(layer, bound)
        # println("after affine")
        # sleep(0.1)
        push!(bounds, nb)
    end
    # println("after for")
    # sleep(0.1)
    reach = ExactReachBound(bounds)
    # reach = ExactReachBound([affine_map(layer, bound) for bound in reach.polys])
    return reach
end

"""
    propagate_layer(prop_method::Box, layer::Dense, reach::LazySet, batch_info)

Propagate the bounds through the dense layer for Ai2 `Box` solver. It operates 
an approximate affine transformation (affine transformation using hyperrectangle 
overapproximation) on the given input bound and returns the output bound. 

## Arguments
- `prop_method` (`Box`): Ai2 `Box` solver used for the verification process.
- `layer` (`Dense`): Dense layer of the model.
- `reach` (`LazySet`): Bound of the input.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `reach` (`hyperrectangle`): Bound of the output after approximate affine 
    transformation.
"""
function propagate_layer(prop_method::Box, layer::Dense, reach::LazySet, batch_info)
    isa(reach, AbstractPolytope) || throw("Ai2 only support AbstractPolytope type branches.")
    reach = approximate_affine_map(layer, reach)
    return reach
end  

"""
    batch_interval_map(W::AbstractMatrix{N}, l::AbstractArray, 
                       u::AbstractArray) where N

Clamps the input to the given bounds and computes the interval map of the 
resulting bound using the given weight matrix.

## Arguments
- `W` (`AbstractMatrix{N}`): Weight matrix of the layer.
- `l` (`AbstractArray`): Lower bound of the input.
- `u` (`AbstractArray`): Upper bound of the input.

## Returns
Tuple of:
- `l_new` (`AbstractArray`): Lower bound of the output.
- `u_new` (`AbstractArray`): Upper bound of the output.
"""
function batch_interval_map(W::AbstractMatrix{N}, l::AbstractArray, u::AbstractArray) where N
    #pos_W = max.(W, fmap(cu, zero(N)))
    #neg_W = min.(W, fmap(cu, zero(N)))
    pos_W = clamp.(W, 0, Inf)
    neg_W = clamp.(W, -Inf, 0)
    l_new = batched_mul(pos_W, l) + batched_mul(neg_W, u) # reach_dim x input_dim+1 x batch
    u_new = batched_mul(pos_W, u) + batched_mul(neg_W, l) # reach_dim x input_dim+1 x batch
    return (l_new, u_new)
end

"""
    propagate_layer_batch(prop_method::Crown, layer::Dense, 
                           bound::CrownBound, batch_info)

Propagates the bounds through the dense layer for `Crown` solver. It operates
an affine transformation on the given input bound and returns the output bound.
It first clamps the input bound and multiplies with the weight matrix using 
`batch_interval_map` function. Then, it adds the bias to the output bound.
The resulting bound is represented by `CrownBound` type.

## Arguments
- `prop_method` (`Crown`): `Crown` solver used for the verification process.
- `layer` (`Dense`): Dense layer of the model.
- `bound` (`CrownBound`): Bound of the input, represented by `CrownBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `new_bound` (`CrownBound`): Bound of the output after affine transformation, 
    which is represented by `CrownBound` type.
"""
function propagate_layer_batch(prop_method::Crown, layer::Dense, bound::CrownBound, batch_info)
    # out_dim x in_dim * in_dim x X_dim x batch_size
    output_Low, output_Up = prop_method.use_gpu ? batch_interval_map(fmap(cu, layer.weight), bound.batch_Low, bound.batch_Up) : batch_interval_map(layer.weight, bound.batch_Low, bound.batch_Up)
    @assert !any(isnan, output_Low) "contains NaN"
    @assert !any(isnan, output_Up) "contains NaN"
    output_Low[:, end, :] .+= prop_method.use_gpu ? fmap(cu, layer.bias) : layer.bias
    output_Up[:, end, :] .+= prop_method.use_gpu ? fmap(cu, layer.bias) : layer.bias
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return new_bound
end

# # Ai2z, Ai2h
# function propagate_layer(prop_method::ForwardProp, layer::Dense, batch_reach::AbstractArray, batch_info)
#     all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
#     batch_reach = identity.(batch_reach) # identity. converts Vector{Any} to Vector{AbstractPolytope}
#     batch_reach = affine_map(layer, batch_reach)
#     return batch_reach
# end

# # Ai2 Box
# function propagate_layer(prop_method::Box, layer::Dense, batch_reach::AbstractArray, batch_info)
#     all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
#     batch_reach = identity.(batch_reach) # identity. converts Vector{Any} to Vector{AbstractPolytope}
#     batch_reach = approximate_affine_map(layer, batch_reach)
#     return batch_reach
# end  

"""
    _preprocess(node, batch_info, bias = nothing)

Preprocesses the bias of the given node for the `BetaCrown` solver. If the bias
is not `nothing`, it multiplies the bias with the beta value of the node.

## Arguments
- `node`: Node of the model.
- `batch_info`: Dictionary containing information of each node in the model.
- `bias`: Bias of the node, default is `nothing`.

## Returns
- `bias`: Preprocessed bias of the node.
"""
function _preprocess(prop_method, node, batch_info, bias = nothing)
    if !isnothing(bias)
        if batch_info[node][:beta] != 1.0 
            bias = prop_method.use_gpu ? fmap(cu, bias) : bias
            node_beta = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta]) : batch_info[node][:beta]
            bias = node_beta .* bias
        end
    end
    return bias
end

"""
    dense_bound_oneside(weight, bias, batch_size)

"""
function dense_bound_oneside(weight, bias, batch_size)
    #weight = reshape(weight, (size(weight)..., 1)) 
    #weight = repeat(weight, 1, 1, batch_size) #add batch dim in weight
    function bound_dense(x)
        if !isnothing(bias)
            return [NNlib.batched_mul(x[1], weight), NNlib.batched_vec(x[1], bias) .+ x[2]]
        else
            return [NNlib.batched_mul(x[1], weight), x[2]]
        end
    end
    return bound_dense
    # if !isnothing(bias)
    #     #bias = reshape(bias, (size(bias)..., 1))
    #     #bias = repeat(bias, 1, batch_size) 
    #     # push!(last_A, x -> [NNlib.batched_mul(x[1], weight), NNlib.batched_vec(x[1], bias) .+ x[2]])
    #     return x -> [NNlib.batched_mul(x[1], weight), NNlib.batched_vec(x[1], bias) .+ x[2]]
    # else
    #     # push!(last_A, x -> [NNlib.batched_mul(x[1], weight), x[2]])
    #     return x -> [NNlib.batched_mul(x[1], weight), x[2]]
    # end
end

"""
    propagate_layer_batch(prop_method::BetaCrown, layer::Dense, 
                           bound::BetaCrownBound, batch_info)

Propagates the bounds through the dense layer for `BetaCrown` solver. It 
operates an affine transformation on the given input bound and returns the
output bound. It first preprocesses the lower- and upper-bounds of the bias of 
the node using `_preprocess`. Then, it computes the interval map of the 
resulting lower- and upper-bounds using `dense_bound_oneside` function. The 
resulting bound is represented by `BetaCrownBound` type.

## Arguments
- `prop_method` (`BetaCrown`): `BetaCrown` solver used for the verification 
    process.
- `layer` (`Dense`): Dense layer of the model.
- `bound` (`BetaCrownBound`): Bound of the input, represented by 
    `BetaCrownBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `New_bound` (`BetaCrownBound`): Bound of the output after affine 
    transformation, which is represented by `BetaCrownBound` type.
"""
function propagate_layer_batch(prop_method::BetaCrown, layer::Dense, bound::BetaCrownBound, batch_info)
    node = batch_info[:current_node]
    #TO DO: we haven't consider the perturbation in weight and bias
    bias_lb = _preprocess(prop_method, node, batch_info, layer.bias)
    bias_ub = _preprocess(prop_method, node, batch_info, layer.bias)
    lA_W = uA_W = nothing 
    # println("=== in dense ===")
    # println("bound.lower_A_x: ", bound.lower_A_x)
    @assert !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
    weight = prop_method.use_gpu ? fmap(cu, layer.weight) : layer.weight
    bias = bias_lb
    lA_x = prop_method.bound_lower ? dense_bound_oneside(weight, bias, batch_info[:batch_size]) : nothing
    uA_x = prop_method.bound_upper ? dense_bound_oneside(weight, bias, batch_info[:batch_size]) : nothing

    New_bound = BetaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return New_bound
    # end
end

function propagate_layer_batch(
    prop_method::MIPVerify,
    layer::Dense,
    bound::AbstractVector,
    batch_info::Dict,
)::AbstractVector
    # create optimization variable of the current node
    node = batch_info[:current_node]
    opt_model = batch_info[:opt_model]
    z = @variable(opt_model, [1:batch_info[node][:size_after_layer][1]])
    batch_info[node][:opt_vars] = Dict(:z => z)

    # get optimization variable of the previous node
    prev_nodes = batch_info[node][:prev_nodes]
    @assert length(prev_nodes) == 1
    z_prev = batch_info[prev_nodes[1]][:opt_vars][:z]

    # add constraint of Dense layer
    @constraint(opt_model, z .== layer.weight * z_prev + layer.bias)

    return bound
end

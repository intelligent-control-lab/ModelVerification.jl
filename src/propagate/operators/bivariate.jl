"""
    propagate_skip(prop_method, layer::typeof(+), bound1::ImageZonoBound, 
                   bound2::ImageZonoBound, batch_info)

Propagate the bounds of the two input layers to the output layer for skip 
connection. The output layer is of type `ImageZonoBound`. The input layers' 
centers and generators are concatenated to form the output layer's center and
generators.

## Arguments
- `prop_method` (`PropMethod`): The propagation method used for the verification 
    problem.
- `layer` (`typeof(+)`): The layer operation to be used for propagation.
- `bound1` (`ImageZonoBound`): The bound of the first input layer.
- `bound2` (`ImageZonoBound`): The bound of the second input layer.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The bound of the output layer represented in `ImageZonoBound` type.
"""
function propagate_skip(prop_method, layer::typeof(+), bound1::ImageZonoBound, bound2::ImageZonoBound, batch_info)
    new_c = bound1.center + bound2.center
    new_g = cat([bound1.generators, bound2.generators]..., dims=4)
    return ImageZonoBound(new_c, new_g)
end

function propagate_skip(prop_method, layer::typeof(+), bound1::Zonotope, bound2::Zonotope, batch_info)
    new_c = bound1.center + bound2.center
    new_g = cat([bound1.generators, bound2.generators]..., dims=2)
    return Zonotope(new_c, new_g)
end

function propagate_skip(prop_method, layer::typeof(+), bound1::Hyperrectangle, bound2::Hyperrectangle, batch_info)
    new_c = LazySets.center(bound1) + LazySets.center(bound2)
    new_r = radius_hyperrectangle(bound1) .+ radius_hyperrectangle(bound2)
    return Hyperrectangle(new_c, new_r)
end


"""
    propagate_skip(prop_method, layer::typeof(+), bound1::ImageStarBound, 
                   bound2::ImageStarBound, batch_info)

Propagate the bounds of the two input layers to the output layer for skip 
connection. The output layer is of type `ImageStarBound`. The input layers' 
centers, generators, and constraints are concatenated to form the output layer's 
center, generators, and constraints.

## Arguments
- `prop_method` (`PropMethod`): The propagation method used for the verification 
problem.
- `layer` (`typeof(+)`): The layer operation to be used for propagation.
- `bound1` (`ImageStarBound`): The bound of the first input layer.
- `bound2` (`ImageStarBound`): The bound of the second input layer.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The bound of the output layer represented in `ImageStarBound` type.
"""
function propagate_skip(prop_method, layer::typeof(+), bound1::ImageStarBound, bound2::ImageStarBound, batch_info)
    new_c = bound1.center + bound2.center
    new_g = cat([bound1.generators, bound2.generators]..., dims=4)
    new_A = cat(bound1.A, bound2.A; dims=(1,2))
    new_b = vcat(bound1.b, bound2.b)
    return ImageStarBound(new_c, new_g, new_A, new_b)
end

# """
#     propagate_skip(prop_method::AlphaCrown, layer::typeof(+), bound1::AlphaCrownBound, bound2::AlphaCrownBound, batch_info)
# """
# function propagate_skip(prop_method::AlphaCrown, layer::typeof(+), bound1::AlphaCrownBound, bound2::AlphaCrownBound, batch_info)
#     New_Lower_A_bias = New_Upper_A_bias = nothing
#     if prop_method.bound_lower
#         New_Lower_A_bias = [Chain(bound1.lower_A_x), Chain(bound2.lower_A_x)]
#     end
#     if prop_method.bound_upper
#         New_Upper_A_bias = [Chain(bound1.upper_A_x), Chain(bound2.upper_A_x)]
#     end
#     return AlphaCrownBound(New_Lower_A_bias, New_Upper_A_bias, nothing, nothing, bound1.batch_data_min, bound1.batch_data_max)
# end
function propagate_skip_batch(prop_method::Crown, layer::typeof(+), bound1::CrownBound, bound2::CrownBound, batch_info)
    @assert bound1.batch_data_max == bound2.batch_data_max
    @assert bound1.batch_data_min == bound2.batch_data_min
    @assert bound1.img_size == bound2.img_size
    # @show size(bound1.batch_Low)
    # @show size([Chain(bound1.batch_Low), Chain(bound2.batch_Low)])
    return CrownBound(bound1.batch_Low+bound2.batch_Low,bound1.batch_Up+bound2.batch_Up,  bound1.batch_data_min, bound1.batch_data_max, bound1.img_size)
end

struct_copy(x::T) where T = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)

+(t1::Tuple, t2::Tuple) = t1 .+ t2
# For backward method, + is not a bivariate operator, The bivariate operator is where the skip starts.
function propagate_layer_batch(prop_method::BetaCrown, layer::typeof(+), bound::BetaCrownBound, batch_info)
    return BetaCrownBound(identity, identity, identity, identity, bound.batch_data_min, bound.batch_data_max, bound.img_size)
end

# For backward method, + is not a bivariate operator, The bivariate operator is where the skip starts.
# There is no node called parallel or skip in ONNX. 
# ONNX directly connect the node with two succeeds to denote a parrallel structure.
# The merging is processed in process_bound -> build_bound_graph -> get_bound_chain
function propagate_skip_batch(prop_method::BetaCrown, layer::typeof(Parallel), bounds::Vector{BetaCrownBound}, batch_info)
    bound = BetaCrownBound(nothing, nothing, nothing, nothing, bounds[1].batch_data_min, bounds[1].batch_data_max, bounds[1].img_size)
    return bound
end

function propagate_skip_batch(
    prop_method::MIPVerify,
    layer::typeof(+),
    bound1::AbstractVector,
    bound2::AbstractVector,
    batch_info,
)::AbstractVector
    # create optimization variable of the current node
    node = batch_info[:current_node]
    opt_model = batch_info[:opt_model]
    z = @variable(opt_model, [1:batch_info[node][:size_after_layer][1]])
    batch_info[node][:opt_vars] = Dict(:z => z)

    # get optimization variable of previous nodes
    prev_nodes = batch_info[node][:prev_nodes]
    @assert length(prev_nodes) == 2
    z_prev1 = batch_info[prev_nodes[1]][:opt_vars][:z]
    z_prev2 = batch_info[prev_nodes[2]][:opt_vars][:z]

    # add constraint of + layer
    @constraint(opt_model, z .== z_prev1 + z_prev2)

    # return any one of the two bounds is fine since it will not be used for verification
    return bound1
end

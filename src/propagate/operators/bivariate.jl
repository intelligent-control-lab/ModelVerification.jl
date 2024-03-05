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

# For backward method, + is not a bivariate operator, The bivariate operator is where the skip starts.
function propagate_layer_batch(prop_method::BetaCrown, layer::typeof(+), bound::BetaCrownBound, batch_info)
    lA_x = deepcopy(bound.lower_A_x)
    uA_x = deepcopy(bound.upper_A_x)
    lA_W = deepcopy(bound.lower_A_W)
    uA_W = deepcopy(bound.upper_A_W)
    op = (+, uuid4())  # use op to denote the position to merge in merge_parallel, 
    # uuid is unique id to make sure the the branch is merged with the right one when there are multiple branches.
    push!(lA_x, op)
    push!(uA_x, op)
    isnothing(lA_W) || push!(lA_W, op)
    isnothing(uA_W) || push!(uA_W, op)
    return BetaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    # return BetaCrownBound(bound.lower_A_x, bound.upper_A_x, bound.lower_A_W, bound.upper_A_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    # return bound
end

function (f::Tuple{typeof(+), Base.UUID})(x)
    return x
end

function merge_parallel(m1::Union{Chain, Vector, Nothing}, m2::Union{Chain, Vector, Nothing})
    if isnothing(m1) && isnothing(m2)
        return nothing
    end
    if length(m1) > length(m2) # make sure m1 is the shorter one
        m1, m2 = m2, m1
    end
    i = findlast(op -> (op isa Tuple) && (op[1] == +), m1) # last_plus_idx
    # merge at the last +, operators before the last + must the same
    return [deepcopy(m1[1:i-1]); [Parallel(+, Chain(deepcopy(m1[i+1:end])), Chain(deepcopy(m2[i+1:end])))]]
end

# For backward method, + is not a bivariate operator, The bivariate operator is where the skip starts.
function propagate_skip_batch(prop_method::BetaCrown, layer::typeof(Parallel), bounds::Vector{BetaCrownBound}, batch_info)
    # @assert bound1.batch_data_max == bound2.batch_data_max
    # @assert bound1.batch_data_min == bound2.batch_data_min
    # @assert bound1.img_size == bound2.img_size
    count_plus = bound -> sum([(op isa Tuple) && (op[1] == +) for op in bound.lower_A_x])
    last_plus_idx = bound -> findlast(op -> (op isa Tuple) && (op[1] == +), bound.lower_A_x)
    last_uuid = bound -> bound.lower_A_x[last_plus_idx(bound)][2]
    bounds = sort(bounds, by = bound -> (count_plus(bound), last_uuid(bound)), rev=true)
    max_plus = maximum(count_plus, bounds)
    if length(bounds) > 3
        @warn "The merging skip method may not work for more than 3 parallel branches."
    end
    # for bound in bounds
    #     @show typeof(bound)
    #     @show typeof(bound.lower_A_x)
    #     @show length(bound.lower_A_x)
    #     [@show op for op in bound.lower_A_x]
    # end
    # lA_x = reduce((b1, b2) -> println(typeof(b2)), bounds)
    lA_x = reduce((b1, b2) -> merge_parallel(b1, b2), [b.lower_A_x for b in bounds])
    uA_x = reduce((b1, b2) -> merge_parallel(b1, b2), [b.upper_A_x for b in bounds])
    lA_W = reduce((b1, b2) -> merge_parallel(b1, b2), [b.lower_A_W for b in bounds])
    uA_W = reduce((b1, b2) -> merge_parallel(b1, b2), [b.upper_A_W for b in bounds])

    # lA_x = merge_parallel(bound1.lower_A_x, bound2.lower_A_x)
    # uA_x = merge_parallel(bound1.upper_A_x, bound2.upper_A_x)
    # lA_W = merge_parallel(bound1.lower_A_W, bound2.lower_A_W)
    # uA_W = merge_parallel(bound1.upper_A_W, bound2.upper_A_W)

    # println("================")
    # for i in eachindex(bound1.lower_A_x)
    #     @show i, typeof(bound1.lower_A_x[i])
    # end
    # println("--------1-------")
    # for i in eachindex(bound2.lower_A_x)
    #     @show i, typeof(bound2.lower_A_x[i])
    # end
    # println("--------2--------")
    # for i in eachindex(lA_x)
    #     @show i, typeof(lA_x[i])
    #     if lA_x[i] isa Parallel
    #         @show length(lA_x[i].layers[1])
    #         @show length(lA_x[i].layers[2])
    #     end
    # end
    # println("================")

    bound = BetaCrownBound(lA_x, uA_x, lA_W, uA_W, deepcopy(bounds[1].batch_data_min), deepcopy(bounds[1].batch_data_max), bounds[1].img_size)
    return bound
end
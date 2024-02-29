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

# For backward method, + is not a bivariate operator, The bivariate operator is where the skip starts.
function propagate_linear_batch(prop_method::BetaCrown, layer::typeof(+), bound::BetaCrownBound, batch_info)
    bound1 = copy(bound)
    bound2 = copy(bound)
    return bound1, bound2
end

function merge(m1::Chain, m2::Chain)
    if length(m1) > length(m2) # make sure m1 is the shorter one
        m1, m2 = m2, m1
    end
    for i in eachindex(m1)
        if m1[i] != m2[i]
            return [m1[1:i-1]; [Parallel(+;m1[i:end], m2[i:end])]]
        elseif i == length(m1) # all are the same -> last node is a skip connection
            return [m1; [SkipConnection(m2[i+1:end], +)]]
        end
    end
end

# For backward method, + is not a bivariate operator, The bivariate operator is where the skip starts.
function propagate_linear_batch(prop_method::BetaCrown, layer::typeof(skip), bound1::BetaCrownBound, bound2::BetaCrownBound, batch_info)
    @assert bound1.batch_data_max == bound2.batch_data_max
    @assert bound1.batch_data_min == bound2.batch_data_min
    @assert bound1.img_size == bound2.img_size
    lA_x = merge(bound1.lA_x, bound2.lA_x)
    uA_x = merge(bound1.uA_x, bound2.uA_x)
    bound = BetaCrownBound(lA_x, uA_x, nothing, nothing, bound1.batch_data_min, bound1.batch_data_max, bound1.img_size)
    return bound
end
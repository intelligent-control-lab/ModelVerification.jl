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
- The bound of the output layer represetned in `ImageZonoBound` type.
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
- The bound of the output layer represetned in `ImageStarBound` type.
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

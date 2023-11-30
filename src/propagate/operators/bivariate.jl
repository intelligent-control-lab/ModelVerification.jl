function propagate_skip(prop_method, layer::typeof(+), bound1::ImageZonoBound, bound2::ImageZonoBound, batch_info)
    new_c = bound1.center + bound2.center
    new_g = cat([bound1.generators, bound2.generators]..., dims=4)
    return ImageZonoBound(new_c, new_g)
end

function propagate_skip(prop_method, layer::typeof(+), bound1::ImageStarBound, bound2::ImageStarBound, batch_info)
    new_c = bound1.center + bound2.center
    new_g = cat([bound1.generators, bound2.generators]..., dims=4)
    new_A = cat(bound1.A, bound2.A; dims=(1,2))
    new_b = vcat(bound1.b, bound2.b)
    return ImageStarBound(new_c, new_g, new_A, new_b)
end

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

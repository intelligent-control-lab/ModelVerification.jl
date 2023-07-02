function forward_skip(prop_method, layer::typeof(+), bound1::ImageZonoBound, bound2::ImageZonoBound, info1, info2)
    new_c = bound1.center + bound2.center
    new_g = cat([bound1.generators, bound2.generators]..., dims=4)
    return ImageZonoBound(new_c, new_g), merge(info1, info2) # if info1 and 2 have the same key, value will be of info2's
end

function forward_skip(prop_method, layer::typeof(+), bound1::ImageStarBound, bound2::ImageStarBound, info1, info2)
    new_c = bound1.center + bound2.center
    new_g = cat([bound1.generators, bound2.generators]..., dims=4)
    new_A = cat(bound1.A, bound2.A; dims=(1,2))
    new_b = vcat(bound1.b, bound2.b)
    return ImageStarBound(new_c, new_g, new_A, new_b), merge(info1, info2) # if info1 and 2 have the same key, value will be of info2's
end

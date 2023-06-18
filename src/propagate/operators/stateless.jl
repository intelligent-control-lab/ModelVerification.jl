forward_linear(prop_method, layer::typeof(flatten), bound::ImageStarBound, batch_info) = 
    [Star(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)), bound.P)], batch_info

forward_linear(prop_method, layer::typeof(flatten), bound::ImageZonoBound, batch_info) =
    [Zonotope(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)))], batch_info

function forward_linear(prop_method, layer::MeanPool, bound::ImageZonoBound, batch_info)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageZonoBound(new_center, new_generators), batch_info
end
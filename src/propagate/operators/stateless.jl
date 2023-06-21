forward_linear(prop_method, layer::typeof(Flux.flatten), bound::ImageStarBound, info) = 
    Star(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)), HPolyhedron(bound.A, bound.b)), info

forward_linear(prop_method, layer::typeof(Flux.flatten), bound::ImageZonoBound, info) =
    Zonotope(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4))), info

function forward_linear(prop_method, layer::MeanPool, bound::ImageStarBound, info)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b), info
end

function forward_linear(prop_method, layer::MeanPool, bound::ImageZonoBound, info)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageZonoBound(new_center, new_generators), info
end
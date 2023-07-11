propagate_linear(prop_method, layer::typeof(flatten), bound::ImageStarBound, batch_info, node::String) = 
    Star(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)), HPolyhedron(bound.A, bound.b))

propagate_linear(prop_method, layer::typeof(flatten), bound::ImageZonoBound, batch_info, node::String) =
    Zonotope(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)))

function propagate_linear(prop_method, layer::MeanPool, bound::ImageStarBound, batch_info, node::String)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b)
end

function propagate_linear(prop_method, layer::MeanPool, bound::ImageZonoBound, batch_info, node::String)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageZonoBound(new_center, new_generators)
end
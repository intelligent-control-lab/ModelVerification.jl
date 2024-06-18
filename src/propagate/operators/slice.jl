
function propagate_layer(prop_method, layer::Slice, bound::Zonotope, batch_info)
    new_center = bound.center[layer.dims...]
    new_generators = bound.generators[layer.dims...,:]
    return Zonotope(new_center, new_generators)
end


function propagate_layer(prop_method, layer::Slice, bound::ImageZonoBound, batch_info)
    new_center = bound.center[layer.dims...]
    new_generators = bound.generators[layer.dims...]
    # @show size(bound.center)
    # @show size(bound.generators)
    # @show layer.dims
    # @show size(new_center)
    # @show size(new_generators)
    return ImageZonoBound(new_center, new_generators)
end

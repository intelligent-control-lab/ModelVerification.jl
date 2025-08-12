
# Copied from NaiveNASflux.jl (onnx_parser/NaiveNASflux/src/NaiveNASflux.jl)
# if we don't use NaiveNASflux, we have to define the Slice layer somewhere our own
struct Slice
    dims::Tuple
end
Flux.@layer Slice
# Define how the layer works on input `x`
(m::Slice)(x::AbstractArray) = x[m.dims...]
function (m::Slice)(x)
    @show x
end


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

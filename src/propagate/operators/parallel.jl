function propagate_parallel_batch(prop_method::ForwardProp, layer, batch_bounds::AbstractArray, batch_info)
    batch_reach_info = [propagate_parallel(prop_method, layer, [b[i] for b in batch_bounds], push!(batch_info, :batch_index => i)) for i in eachindex(batch_bounds[1])]
    return batch_reach_info#map(first, batch_reach_info)
end

function propagate_parallel(prop_method, layer::typeof(+), bounds::Vector{<:ImageZonoBound}, batch_info)
    new_c = reduce(+, [b.center for b in bounds])
    new_g = cat([b.generators for b in bounds]..., dims=4)
    return ImageZonoBound(new_c, new_g)
end

function propagate_parallel(prop_method, layer::typeof(+), bounds::Vector{<:LazySets.Zonotope}, batch_info)
    new_c = reduce(+, [b.center for b in bounds])
    new_g = cat([b.generators for b in bounds]..., dims=2)
    return Zonotope(new_c, new_g)
end

function propagate_parallel(prop_method, layer::typeof(vcat), bounds::Vector{<:ImageZonoBound}, batch_info)
    new_c = cat([b.center for b in bounds]..., dims=1)
    new_g = cat([b.generators for b in bounds]..., dims=1)
    return ImageZonoBound(new_c, new_g)
end

function propagate_parallel(prop_method, layer::typeof(vcat), bounds::Vector{<:LazySets.Zonotope}, batch_info)
    new_c = cat([b.center for b in bounds]..., dims=1)
    new_g = cat([b.generators for b in bounds]..., dims=1)
    return Zonotope(new_c, new_g)
end

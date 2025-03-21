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
    # new_c = cat([b.center for b in bounds]..., dims=1)
    # new_g = cat([b.generators for b in bounds]..., dims=1)
    # println("new_g", new_g)
    # return Zonotope(new_c, new_g)

    total_dim = sum(length(z.center) for z in bounds)
    
    # Compute the new center by vertically stacking the centers
    new_center = vcat([z.center for z in bounds]...)
    
    # Compute the new generator matrix
    total_generators = sum(size(z.generators, 2) for z in bounds)
    new_generators = zeros(total_dim, total_generators)
    
    row_offset = 1
    col_offset = 1
    for z in bounds
        rows = size(z.generators, 1)
        cols = size(z.generators, 2)
        new_generators[row_offset:row_offset+rows-1, col_offset:col_offset+cols-1] = z.generators
        row_offset += rows
        col_offset += cols
    end
    
    return Zonotope(new_center, new_generators)
end

function forward_act(prop_method, layer::typeof(relu), bound::ImageZonoBound, batch_info)
    cen = reshape(bound.center, :)
    gen = reshape(bound.generators, :, size(bound.generators,4))
    flat_reach = overapproximate(Rectification(Zonotope(cen, gen)), Zonotope)
    new_cen = reshape(center(flat_reach), size(bound.center))
    sz = size(bound.generators)
    println("before size: ", sz)
    new_gen = reshape(genmat(flat_reach), sz[1], sz[2], sz[3], :)
    println("after size: ", size(new_gen))
    new_bound = ImageZonoBound(new_cen, new_gen)
    return new_bound, batch_info
end  


function forward_act(prop_method, layer::typeof(relu), bound::CrownBound, batch_info)
    
    output_Low, output_Up = copy(bound.batch_Low), copy(bound.batch_Up) # reach_dim x input_dim x batch

    # If the lower bound of the lower bound is positive,
    # No change to the linear bounds.
    
    # If the upper bound of the upper bound is negative, set
    # both linear bounds to 0
    l, u = compute_bound(bound) # reach_dim x batch
    
    inact_mask = u .<= 0 # reach_dim x batch
    inact_mask_ext = broadcast_mid_dim(inact_mask, output_Low) # reach_dim x input_dim x batch
    output_Low[inact_mask_ext] .= 0
    output_Up[inact_mask_ext] .= 0

    
    # if the bounds overlap 0, concretize by setting
    # the generators to 0, and setting the new upper bound
    # center to be the current upper-upper bound.
    unstable_mask = (u .> 0) .& (l .< 0) # reach_dim x batch
    unstable_mask_ext = broadcast_mid_dim(unstable_mask, output_Low) # reach_dim x input_dim+1 x batch
    slope = u[unstable_mask] ./ (u[unstable_mask] .- l[unstable_mask]) # selected_reach_dim * selected_batch
    slope_mtx = ones(size(u))

    slope_mtx[unstable_mask] = u[unstable_mask] ./ (u[unstable_mask] .- l[unstable_mask]) # reach_dim x batch
    broad_slope = broadcast_mid_dim(slope_mtx, output_Up) # selected_reach_dim x input_dim+1 x selected_batch
    # broad_slop = reshape(slope, )
    output_Up .*= broad_slope
    unstable_mask_bias = copy(unstable_mask_ext)
    unstable_mask_bias[:,1:end-1,:] .= 0
    output_Up[unstable_mask_bias] .+= (slope .* max.(-u[unstable_mask], 0))[:]

    # output_Low[unstable_mask_ext] .*= broad_slope[:]
    output_Low[unstable_mask_ext] .= 0

    @assert !any(isnan, output_Low) "relu low contains NaN"
    @assert !any(isnan, output_Up) "relu up contains NaN"
    
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max)
    l, u = compute_bound(new_bound)

    return new_bound, batch_info
end

function forward_act(prop_method::Ai2h, layer::typeof(relu), batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [convex_hull(UnionSetArray(forward_partition(layer, reach))) for reach in batch_reach]
    return batch_reach, batch_info
end

function forward_act(prop_method::Union{Ai2z, ImageStarZono}, layer::typeof(relu), batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [overapproximate(Rectification(reach), Zonotope) for reach in batch_reach]
    return batch_reach, batch_info
end  

function forward_act(prop_method::Box, layer::typeof(relu), batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [rectify(reach) for reach in batch_reach]
    return batch_reach, batch_info
end  

function forward_partition(layer::typeof(relu), reach)
    N = dim(reach)
    output = HPolytope{Float64}[]
    for h in 0:(2^N)-1
        P = Diagonal(1.0.*digits(h, base = 2, pad = N))
        orthant = HPolytope(Matrix(I - 2.0P), zeros(N))
        S = intersection(reach, orthant)
        if !isempty(S)
            push!(output, linear_map(P, S))
        end
    end
    return output
end
    
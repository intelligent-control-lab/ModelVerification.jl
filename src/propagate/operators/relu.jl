
function forward_act(prop_method::Union{Ai2z, ImageStarZono}, layer::typeof(relu), reach::AbstractPolytope, info)
    reach = overapproximate(Rectification(reach), Zonotope)
    return reach, info
end  

function forward_act(prop_method::Ai2h, layer::typeof(relu), reach::AbstractPolytope, info)
    reach = convex_hull(UnionSetArray(forward_partition(layer, reach)))
    return reach, info
end

function forward_act(prop_method::Box, layer::typeof(relu), reach::AbstractPolytope, info)
    reach = rectify(reach)
    return reach, info
end  

function forward_act(prop_method, layer::typeof(relu), bound::ImageZonoBound, info)
    cen = reshape(bound.center, :)
    gen = reshape(bound.generators, :, size(bound.generators,4))
    flat_reach = overapproximate(Rectification(Zonotope(cen, gen)), Zonotope)
    new_cen = reshape(center(flat_reach), size(bound.center))
    sz = size(bound.generators)
    # println("before size: ", sz)
    new_gen = reshape(genmat(flat_reach), sz[1], sz[2], sz[3], :)
    # println("after size: ", size(new_gen))
    new_bound = ImageZonoBound(new_cen, new_gen)
    return new_bound, info
end

function forward_act(prop_method, layer::typeof(relu), bound::Star, info)
    cen = center(bound) # h * w * c * 1
    gen = basis(bound) # h*w*c x n_alpha
    n_con = length(constraints_list(bound.P))
    n_alpha = size(gen, 2)
    box = overapproximate(bound, Hyperrectangle)
    l, u = low(box), high(box)
    
    bA = permutedims(cat([con.a for con in constraints_list(bound.P)]..., dims=2)) # n_con x n_alpha
    bb = vcat([con.b for con in constraints_list(bound.P)]...) # n_con
    
    slope = u ./ (u-l)
    unstable_mask = (u .> 0) .& (l .< 0) # hwc
    slope = u[unstable_mask] ./ (u[unstable_mask] .- l[unstable_mask]) # n_beta
    n_beta = sum(unstable_mask)
    indices = findall(unstable_mask)
    # beta_gen = sparse(1:length(indices), indices, 1)
    # beta_gen = permutedims(Matrix(Flux.onehot.(indices, 1:length(u)))) # hwc * n_beta
    beta_gen = zeros(length(u), n_beta)
    beta_gen[CartesianIndex.(indices, 1:length(indices))] .= 1

    # beta >= 0  ->  -beta <= 0
    A1_beta = [zeros(n_beta, n_alpha) Matrix(-1.0I, n_beta, n_beta)] 
    b1_beta = zeros(n_beta)
    # beta >= x  ->  beta >= cen + alpha*gen  ->  alpha*gen - beta<= -cen
    A2_beta = [gen[unstable_mask,:] Matrix(-1.0I, n_beta, n_beta)] 
    b2_beta = -cen[unstable_mask]
    # beta <= (x-l)*u/(u-l) -> beta <= (x-l)*k ->  beta <= (cen + alpha*gen - l)*k  ->
    # -k*alpha*gen + beta <=  k*(cen - l)
    A3_beta = [.-slope .* gen[unstable_mask,:] Matrix(1.0I, n_beta, n_beta)] 
    b3_beta = slope .* (cen[unstable_mask] - l[unstable_mask])

    A = [bA zeros(n_con, n_beta);
        A1_beta;
        A2_beta;
        A3_beta]
    b = [bb; b1_beta; b2_beta; b3_beta]

    cen[unstable_mask] .= 0
    gen[unstable_mask, :] .= 0

    T = eltype(cen)
    new_bound = Star(T.(cen), T.([gen beta_gen]), HPolyhedron(T.(A),T.(b)))
    return new_bound, info
end  

function ImageStar_to_Star(bound::ImageStarBound)
    cen = reshape(bound.center, :) # h * w * c * 1
    gen = reshape(bound.generators, :, size(bound.generators,4)) # h*w*c x n_alpha
    T = eltype(cen)
    return Star(T.(cen), T.(gen), HPolyhedron(T.(bound.A), T.(bound.b)))
end

function Star_to_ImageStar(bound::Star, sz)
    new_cen = reshape(center(bound), sz[1], sz[2], sz[3], 1)
    new_gen = reshape(basis(bound), sz[1], sz[2], sz[3], :) # h x w x c x (n_alpha + n_beta)
    A = permutedims(cat([con.a for con in constraints_list(bound.P)]..., dims=2)) # n_con x n_alpha
    b = vcat([con.b for con in constraints_list(bound.P)]...) # n_con
    T = eltype(new_cen)
    return ImageStarBound(T.(new_cen), T.(new_gen), T.(A), T.(b))
end

function forward_act(prop_method, layer::typeof(relu), bound::ImageStarBound, info)
    sz = size(bound.generators)
    flat_bound = ImageStar_to_Star(bound)
    new_flat_bound, info = forward_act(prop_method, layer, flat_bound, info)
    new_bound = Star_to_ImageStar(new_flat_bound, sz)
    return new_bound, info
end

function forward_act_batch(prop_method, layer::typeof(relu), bound::CrownBound, batch_info)
    
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

    output_Up[unstable_mask_bias] .+= (slope .* max.(-l[unstable_mask], 0))[:]

    # output_Low[unstable_mask_ext] .*= broad_slope[:]
    output_Low[unstable_mask_ext] .= 0

    @assert !any(isnan, output_Low) "relu low contains NaN"
    @assert !any(isnan, output_Up) "relu up contains NaN"
    
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max)
    return new_bound, batch_info
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
    
"""
    propagate_act(prop_method::Union{Ai2z, ImageZono}, layer::typeof(relu), 
                  reach::AbstractPolytope, batch_info)

Propagate the `AbstractPolytope` bound through a ReLU layer. I.e., it applies 
the ReLU operation to the `AbstractPolytope` bound. The resulting bound is also
of type `AbstractPolytope`. This is for either `Ai2z` or `ImageZono` propagation 
methods, which both use Zonotope-like representation for the safety 
specifications. After rectifying the input bound, it overapproximates the 
resulting bound using a Zonotope.

## Arguments
- `prop_method` (`Union{Ai2z, ImageZono}`): The propagation method used for the 
    verification problem. It can be either `Ai2z` or `ImageZono`, which both use 
    Zonotope-like representation for the safety specifications.
- `layer` (`typeof(relu)`): The ReLU operation to be used for propagation.
- `reach` (`AbstractPolytope`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- the relued bound of the output represented in `Zonotope` type.
"""
function propagate_act(prop_method::Union{Ai2z, ImageZono}, layer::typeof(relu), reach::AbstractPolytope, batch_info)
    reach = overapproximate(Rectification(reach), Zonotope)
    return reach
end  

"""
    partition_relu(bound)

Partition the `bound` into multiple `VPolytope` objects, each of which is the 
intersection of the `bound` and an orthant. The resulting `VPolytope` objects 
are stored in an array. This is for ReLU propagations in `ExactReach` solver.
Thus, the resulting `VPolytope` objects are the outputs of rectifying the input 
bound. The dimension of the `bound` must be less than 30, since otherwise the 
number of output sets will be too large.

## Arguments
- `bound`: The bound of the input node.

## Returns
- An array of partitioned bounds represented using `VPolytope` type.
"""
function partition_relu(bound)
    N = dim(bound)
    N > 30 && @warn "Got dim(X) == $N in `forward_partition`. Expecting 2ᴺ = $(2^big(N)) output sets."
    output = []
    cnt = 0
    for h in 0:(big"2"^N)-1
        cnt += 1
        P = Diagonal(1.0.*digits(h, base = 2, pad = N))
        orthant = HPolytope(Matrix(I - 2.0P), zeros(N))
        S = intersection(bound, orthant)
        if !isempty(S)
            squeezed = VPolytope([P*v for v in vertices_list(S)])
            length(squeezed.vertices) <= 1 && continue # no need to keep single points, because it must lie on a line.
            # squeezed = linear_map(P, S))
            # squeezed = linear_map(P, S)
            push!(output, squeezed)
        end
    end
    return output
end

"""
    propagate_act(prop_method::ExactReach, layer::typeof(relu), 
                  reach::ExactReachBound, batch_info)

Propagate the `ExactReachBound` bound through a ReLU layer. I.e., it applies 
the ReLU operation to the `ExactReachBound` bound. The resulting bound is also 
of type `ExactReachBound`. This is for `ExactReach` propagation method.
It calls `partition_relu` that partitions the resulting rectified bound into 
multiple `VPolytope` objects, each of which is the intersection of the resulting 
bound and an orthant. The resulting `VPolytope` objects are vertically 
concatenated and stored in an `ExactReachBound` object.

## Arguments
- `prop_method` (`ExactReach`): The propagation method used for the verification 
    problem.
- `layer` (`typeof(relu)`): The ReLU operation to be used for propagation.
- `reach` (`ExactReachBound`): The bound of the input node, represented using 
    `ExactReachBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- the relued bound of the output represented in `ExactReachBound` type.
"""
function propagate_act(prop_method::ExactReach, layer::typeof(relu), reach::ExactReachBound, batch_info)
    partitioned_bound = [partition_relu(bound) for bound in reach.polys]
    partitioned_bound = vcat(partitioned_bound...)
    reach = ExactReachBound(partitioned_bound)
    return reach
end

"""
    propagate_act(prop_method::Box, layer::typeof(relu), 
                  reach::AbstractPolytope, batch_info)

Propagate the `AbstractPolytope` bound through a ReLU layer. I.e., it applies 
the ReLU operation to the `AbstractPolytope` bound. The resulting bound is also 
of type `AbstractPolytope`. This is for Ai2's `Box` propagation method. It calls 
`rectify` that rectifies the input bound.

## Arguments
- `prop_method` (`Box`): The propagation method used for the verification 
    problem.
- `layer` (`typeof(relu)`): The ReLU operation to be used for propagation.
- `reach` (`AbstractPolytope`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- the relued bound of the output represented in `AbstractPolytope` type.
"""
function propagate_act(prop_method::Box, layer::typeof(relu), reach::AbstractPolytope, batch_info)
    reach = rectify(reach)
    return reach
end  

"""
    fast_overapproximate(r::Rectification{N,<:AbstractZonotope}, 
                         ::Type{<:Zonotope}) where {N}

Computes the overapproximation of the rectified set `r` using a Zonotope.

## Arguments
- `r` (`Rectification`): The rectified set.
- `::Type{<:Zonotope}`: The type of the overapproximation, default is 
    `Zonotope`.

## Returns
- The overapproximation of the rectified set `r` using a Zonotope.
"""
function fast_overapproximate(r::Rectification{N,<:AbstractZonotope}, ::Type{<:Zonotope}) where {N}
    Z = LazySets.set(r)     # Returns the original set of the rectification.
    c = copy(LazySets.center(Z))
    G = copy(LazySets.genmat(Z))
    n, m = size(G)

    # stats = @timed l, u = low(Z), high(Z)
    l, u = compute_bound(Z) # Computes lower- and upper-bounds of the original set.
    # println("non0 ele cnt: ", sum((u - l) .> 1e-8))
    # println("low high time: ", stats.time)
    # println(l)
    # mask_activate = l .> 0
    mask_inactivate = u .<= 0
    mask_unstable = (l .< 0) .& (u .> 0)
    c[mask_inactivate] .= zero(N)
    G[mask_inactivate,:] .= zero(N)
    
    λ = u[mask_unstable] ./ (u[mask_unstable] .- l[mask_unstable]) # n_unstable
    μ = λ .* l[mask_unstable] ./ -2 # n_unstable
    
    c[mask_unstable] = c[mask_unstable] .* λ .+ μ
    G[mask_unstable,:] = G[mask_unstable,:] .* λ

    q = sum(mask_unstable)
    if q >= 1
        Gnew = zeros(N, n, q)
        indices = findall(mask_unstable)
        Gnew[CartesianIndex.(indices, 1:q)] .= μ
        Gout = hcat(G, Gnew)
    else 
        Gout = G
    end
    
    return Zonotope(c, LazySets.remove_zero_columns(Gout))
end

"""
    propagate_act(prop_method, layer::typeof(relu), 
                  bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a ReLU layer. I.e., it applies 
the ReLU operation to the `ImageZonoBound` bound. The resulting bound is also 
of type `ImageZonoBound`. This is for `ImageZono` propagation method. It 
flattens the input bound into a `Zonotope` and calls `fast_overapproximate` that 
computes the overapproximation of the rectified set using a Zonotope. It then 
converts the resulting `Zonotope` back to `ImageZonoBound`.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(relu)`): The ReLU operation to be used for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node, represented using 
    `ImageZonoBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- the relued bound of the output represented in `ImageZonoBound` type.
"""
function propagate_act(prop_method, layer::typeof(relu), bound::ImageZonoBound, batch_info)
    cen = reshape(bound.center, :)
    gen = reshape(bound.generators, :, size(bound.generators,4))
    # println("size gen: ", size(bound.generators,4))
    flat_reach = Zonotope(cen, gen)
    # println("before order: ", float(order(flat_reach)))
    # sleep(0.1)
    stats = @timed flat_reach = fast_overapproximate(Rectification(flat_reach), Zonotope)
    # println("overapproximate time: ", stats.time)
    # sleep(0.1)
    # flat_reach = overapproximate(Rectification(flat_reach), Zonotope)
    # diff = LazySets.center(fast_reach) - LazySets.center(flat_reach)
    # println(diff[1:10])
    # println(findall(diff != 0))
    # @assert all(LazySets.center(fast_reach) ≈ LazySets.center(flat_reach))
    # @assert LazySets.genmat(fast_reach) == LazySets.genmat(flat_reach)
    # flat_reach = box_approximation(Rectification(flat_reach))
    # println("after order: ", float(order(flat_reach)))
    # sleep(0.1)
    # stats = @timed flat_reach = remove_redundant_generators(flat_reach)
    # println("remove redundant time: ", stats.time)
    # println("after reducing order: ", float(order(flat_reach)))
    # sleep(0.1)
    if size(genmat(flat_reach),2) > 10
        # println("before reducing order: ", float(order(flat_reach)))
        flat_reach = remove_redundant_generators(flat_reach)
        # println("after reducing order:  ", float(order(flat_reach)))
    end
    new_cen = reshape(LazySets.center(flat_reach), size(bound.center))
    sz = size(bound.generators)
    # println("before size: ", sz)
    new_gen = reshape(genmat(flat_reach), sz[1], sz[2], sz[3], :)
    # println("after size: ", size(new_gen))
    new_bound = ImageZonoBound(new_cen, new_gen)
    return new_bound
end

"""
    propagate_act(prop_method, layer::typeof(relu), bound::Star, batch_info)
    
Propagate the `Star` bound through a ReLU layer. I.e., it applies the ReLU 
operation to the `Star` bound. The resulting bound is also of type `Star`. This 
is for `Star` propagation methods.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(relu)`): The ReLU operation to be used for propagation.
- `bound` (`Star`): The bound of the input node, represented using `Star` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- the relued bound of the output represented in `Star` type.

## Reference
[1] HD. Tran, S. Bak, W. Xiang, and T.T. Johnson, "Verification of Deep Convolutional 
Neural Networks Using ImageStars," in _Computer Aided Verification (CAV)_, 2020.
"""
function propagate_act(prop_method, layer::typeof(relu), bound::Star, batch_info)
    # https://arxiv.org/pdf/2004.05511.pdf
    cen = LazySets.center(bound) # h * w * c * 1
    gen = basis(bound) # h*w*c x n_alpha
    n_con = length(constraints_list(bound.P))
    n_alpha = size(gen, 2)
    l, u = nothing, nothing
    to = get_timer("Shared")
    if hasproperty(prop_method, :pre_bound_method) && !isnothing(prop_method.pre_bound_method)
        node = batch_info[:current_node]
        batch_index = batch_info[:batch_index]
        l, u = compute_bound(batch_info[node][:pre_bound][batch_index])
        l = reshape(l, size(cen))
        u = reshape(u, size(cen))
    else
        @timeit to "over_approx" box = overapproximate(bound, Hyperrectangle)
        l, u = low(box), high(box)
    end
    
    bA = permutedims(cat([con.a for con in constraints_list(bound.P)]..., dims=2)) # n_con x n_alpha
    bb = vcat([con.b for con in constraints_list(bound.P)]...) # n_con
    
    
    inactive_mask = u .<= 0

    cen[inactive_mask] .= 0
    gen[inactive_mask, :] .= 0

    active_mask = l .>= 0
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
    # display(plot(new_bound, xlim=[-3,3], ylim=[-3,3], title=string(typeof(prop_method))*" after relu"))
    return new_bound
end  

"""
    ImageStar_to_Star(bound::ImageStarBound)

Convert the `ImageStarBound` bound to `Star` bound.

## Arguments
- `bound` (`ImageStarBound`): The bound of the input node, represented using 
    `ImageStarBound` type.

## Returns
- The bound represented using `Star` type.
"""
function ImageStar_to_Star(bound::ImageStarBound)
    cen = reshape(bound.center, :) # h * w * c * 1
    gen = reshape(bound.generators, :, size(bound.generators,4)) # h*w*c x n_alpha
    T = eltype(cen)
    return Star(T.(cen), T.(gen), HPolyhedron(T.(bound.A), T.(bound.b)))
end

"""
    Star_to_ImageStar(bound::Star, sz)

Converts the `Star` bound to `ImageStarBound` bound.

## Arguments
- `bound` (`Star`): The bound of the input node, represented using `Star` type.
- `sz`: The size of the input image, i.e., the target size.

## Returns
- The bound represented using `ImageStarBound` type.
"""
function Star_to_ImageStar(bound::Star, sz)
    new_cen = reshape(LazySets.center(bound), sz[1], sz[2], sz[3], 1)
    new_gen = reshape(basis(bound), sz[1], sz[2], sz[3], :) # h x w x c x (n_alpha + n_beta)
    A = permutedims(cat([con.a for con in constraints_list(bound.P)]..., dims=2)) # n_con x n_alpha
    b = vcat([con.b for con in constraints_list(bound.P)]...) # n_con
    T = eltype(new_cen)
    return ImageStarBound(T.(new_cen), T.(new_gen), T.(A), T.(b))
end

"""
    propagate_act(prop_method, layer::typeof(relu), 
                  bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a ReLU layer. I.e., it applies 
the ReLU operation to the `ImageStarBound` bound. The resulting bound is also 
of type `ImageStarBound`. This is for `ImageStar` propagation method. It 
converts the input bound to `Star` type, calls `propagate_act` that propagates 
the `Star` bound through a ReLU layer, and converts the resulting bound back to 
`ImageStarBound`.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(relu)`): The ReLU operation to be used for propagation.
- `bound` (`ImageStarBound`): The bound of the input node, represented using 
    `ImageStarBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The relued bound of the output represented in `ImageStarBound` type.
"""
function propagate_act(prop_method, layer::typeof(relu), bound::ImageStarBound, batch_info)
    to = get_timer("Shared")
    sz = size(bound.generators)
    # println("generator size: ", sz)
    @timeit to "ImageStar_to_Star" flat_bound = ImageStar_to_Star(bound)
    @timeit to "propagate_star" new_flat_bound = propagate_act(prop_method, layer, flat_bound, batch_info)
    @timeit to "Star_to_ImageStar" new_bound = Star_to_ImageStar(new_flat_bound, sz)
    return new_bound
end

"""
    propagate_act_batch(prop_method::Crown, layer::typeof(relu), 
                        bound::CrownBound, batch_info)

Propagate the `CrownBound` bound through a ReLU layer.
"""
function propagate_act_batch(prop_method::Crown, layer::typeof(relu), original_bound::CrownBound, batch_info)
    to = get_timer("Shared")
    if length(size(original_bound.batch_Low)) > 3
        bound, img_size = convert_CROWN_Bound_batch(original_bound)
    else
        bound = original_bound
    end
    output_Low, output_Up = copy(bound.batch_Low), copy(bound.batch_Up) # reach_dim x input_dim x batch


    # If the lower bound of the lower bound is positive,
    # No change to the linear bounds.
    
    # If the upper bound of the upper bound is negative, set
    # both linear bounds to 0
    @timeit to "compute_bound"  l, u = compute_bound(bound) # reach_dim x batch
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

    slope_mtx = prop_method.use_gpu ? fmap(cu, ones(size(u))) : ones(size(u))
    if prop_method.use_gpu
        CUDA.@allowscalar slope_mtx[unstable_mask] = u[unstable_mask] ./ (u[unstable_mask] .- l[unstable_mask]) # reach_dim x batch
    else
        slope_mtx[unstable_mask] = u[unstable_mask] ./ (u[unstable_mask] .- l[unstable_mask])
    end

    broad_slope = broadcast_mid_dim(slope_mtx, output_Up) # selected_reach_dim x input_dim+1 x selected_batch
    # broad_slop = reshape(slope, )
    output_Up .*= broad_slope
    unstable_mask_bias = copy(unstable_mask_ext)
    unstable_mask_bias[:,1:end-1,:] .= 0

    if prop_method.use_gpu
        CUDA.@allowscalar output_Up[unstable_mask_bias] .+= (slope .* max.(-l[unstable_mask], 0))[:]
    else
        output_Up[unstable_mask_bias] .+= (slope .* max.(-l[unstable_mask], 0))[:]
    end

    # output_Low[unstable_mask_ext] .*= broad_slope[:]
    output_Low[unstable_mask_ext] .= 0

    @assert !any(isnan, output_Low) "relu low contains NaN"
    @assert !any(isnan, output_Up) "relu up contains NaN"
    
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    
    if length(size(original_bound.batch_Low)) > 3
        new_bound = convert_CROWN_Bound_batch(new_bound, img_size)
    end
    # @show size(new_bound.batch_Low)
    return new_bound
end
#initalize relu's alpha_lower and alpha_upper
 

#= A spec x reach x batch
   S        reach x batch
beta        reach x batch
A .+ S.* beta =#

mutable struct BetaLayer
    node
    alpha
    beta
    beta_S
    beta_index
    spec_A_b
    lower
    unstable_mask
    active_mask 
    upper_slope
    lower_bias
    upper_bias
end
Flux.@functor BetaLayer (alpha, beta,) #only alpha/beta need to be trained

"""
    relu_upper_bound(lower, upper)

Compute the upper bound slope and intercept according to CROWN relaxation. 

## Arguments
- `lower`: The lower bound of the input node, pre-ReLU operation.
- `upper`: The upper bound of the input node, pre-ReLU operation.

## Returns
- The upper bound slope and intercept according to CROWN relaxation.
"""
function relu_upper_bound(lower, upper)
    lower_r = clamp.(lower, -Inf, 0)
    upper_r = clamp.(upper, 0, Inf)
    upper_r .= max.(upper_r, lower_r .+ 1e-8)
    upper_slope = upper_r ./ (upper_r .- lower_r) #the slope of the relu upper bound
    upper_bias = - lower_r .* upper_slope #the bias of the relu upper bound
    return upper_slope, upper_bias
end

#using last_A for getting New_A
"""
    multiply_by_A_signs(last_A, slope_pos, slope_neg)

Multiply the last layer's activation by the sign of the slope of the ReLU 
activation function. This is for `BetaLayer` propagation method. 
"""
function multiply_by_A_signs(last_A, slope_pos, slope_neg)
    #last_A : spec_dim x reach_dim x batch_dim
    #slope_pos : reach_dim x batch_dim
    A_pos = clamp.(last_A, 0, Inf)
    A_neg = clamp.(last_A, -Inf, 0)
    if ndims(slope_pos) != 1 # Special case for LSTM when bias term is 1-dimension. 
        slope_pos = repeat(reshape(slope_pos,(1, size(slope_pos)...)), size(A_pos)[1], 1, 1) #add spec dim for slope_pos
        slope_neg = repeat(reshape(slope_neg,(1, size(slope_neg)...)), size(A_neg)[1], 1, 1) #add spec dim for slope_pos
    end
    # println("A_pos: ", A_pos)
    # println("A_neg: ", A_neg)
    # println("slope_pos: ", slope_pos)
    # println("slope_neg: ", slope_neg)
    New_A = slope_pos .* A_pos .+ slope_neg .* A_neg 
    return New_A
end

function multiply_bias(last_A, bias_pos, bias_neg)
    # println("last_A:   ", last_A)
    # println("bias_pos: ", bias_pos)
    # println("bias_neg: ", bias_neg)
    #last_A : spec_dim x reach_dim x batch_dim
    #bias_pos : reach_dim x batch_dim
    A_pos = clamp.(last_A, 0, Inf)
    A_neg = clamp.(last_A, -Inf, 0) 
    if isnothing(bias_pos)
        return NNlib.batched_vec(A_neg, bias_neg)
    elseif isnothing(bias_neg)
        return NNlib.batched_vec(A_pos, bias_pos)
    end
    new_b = NNlib.batched_vec(A_pos, bias_pos) .+ NNlib.batched_vec(A_neg, bias_neg)
    # println("new_b pos: ", NNlib.batched_vec(A_pos, bias_pos))
    # println("new_b neg: ", NNlib.batched_vec(A_neg, bias_neg))
    # println("new_b: ", new_b)
    return new_b
end

#bound oneside of the relu, like upper or lower
"""
    bound_oneside(last_A, slope_pos, slope_neg)

Bound the ReLU activation function from one side, such as upper or lower.

## Arguments
- `last_A`: The last layer's activation.
- `slope_pos`: The slope of the ReLU activation function from the positive side.
- `slope_neg`: The slope of the ReLU activation function from the negative side.

## Returns
- The bound of the ReLU activation function from one side.
"""
function bound_oneside(last_A, slope_pos, slope_neg)
    if isnothing(last_A)
        return nothing
    end
    New_A = multiply_by_A_signs(last_A, slope_pos, slope_neg)
    return New_A
end


function add_beta(A, beta, beta_S)
    #buffer_beta = Zygote.Buffer(beta)
    #original_size_beta = original_size_beta .* beta_S
    beta_split = clamp.(beta, 0, Inf) .* beta_S
    # println("beta: ", beta)
    # println("beta_S: ", beta_S)
    # println("beta_split: ", beta_split)
    # println("size(beta): ", size(beta))
    # println("size(beta_S): ", size(beta_S))
    # New_A = A .+ NNlib.batched_mul(spec_A_b[1], reshape(original_size_beta, (1, size(original_size_beta)...)))
    # println("add beta")
    # println("size(A): ", size(A))
    # println(size(beta))
    # println(size(beta_S))
    # println(size(beta_split))
    # println("size(beta_split): ", size(reshape(beta_split, (1, size(beta_split)...))))
    New_A = A .+ reshape(beta_split, (1, size(beta_split)...)) # NNlib.batched_mul(spec_A_b[1], reshape(beta_split, (1, size(beta_split)...)))
    return New_A
end

function vecbeta_convert_to_original_size(index, vector, original)
    original_size_matrix = zeros(size(vec(original)))
    if !isnothing(index)
        original_size_matrix[index] .= vector
    end
    original_size_matrix = reshape(original_size_matrix, size(original)..., 1)
    return original_size_matrix
end

function convert_vec_beta_to_original_size(beta, beta_S, beta_index)
    original_size_beta = cat(vecbeta_convert_to_original_size(beta_index[i], beta[i], beta_S[i]) for i in eachindex(beta), dims = ndims(beta_S)) 
    return original_size_beta
end

function (f::BetaLayer)(x)
    # to = get_timer("Shared")
    
    A = x[1]
    b = x[2]
    # println("A: ", A)
    # println("b: ", b)
    # @assert !any(isnan.(A))
    if isnothing(A)
        return [nothing, nothing]
    end
    # @timeit to "beta_layer" begin
    # lower_slop = alpha if unstable, 1 if active, 0 if inactive
    lower_slope = clamp.(f.alpha, 0, 1) .* f.unstable_mask .+ f.active_mask 
    if f.lower 
        New_b = multiply_bias(A, f.lower_bias, f.upper_bias) .+ b
        # println("lower New_b: ", New_b)
        # println("lower_slope: ", lower_slope)
        # println("f.upper_slope: ", f.upper_slope)
        New_A = bound_oneside(A, lower_slope, f.upper_slope)
        # println("lower New_A: ", New_A)
        # println("f.beta: ", f.beta)
        # @assert !any(isnan.(f.beta))
        New_A = add_beta(New_A, f.beta, f.beta_S)
        # println("lower New_b: ", New_b)
        # println("lower New_A: ", New_A)
        
    else
        New_b = multiply_bias(A, f.upper_bias, f.lower_bias) .+ b
        # println("upper New_b: ", New_b)
        New_A = bound_oneside(A, f.upper_slope, lower_slope)
        New_A = add_beta(New_A, f.beta, f.beta_S)
        # println("upper New_A: ", New_A)
        
    end
    # end
    return [New_A, New_b]
end

function propagate_act_batch(prop_method::BetaCrown, layer::typeof(relu), bound::BetaCrownBound, batch_info)
    node = batch_info[:current_node]
    #= if !haskey(batch_info[node], :pre_lower) || !haskey(batch_info[node], :pre_upper)
        lower, upper = compute_bound(batch_info[node][:pre_bound])
        batch_info[node][:pre_lower] = lower
        batch_info[node][:pre_upper] = upper
    else =#
    lower = batch_info[node][:pre_lower]
    upper = batch_info[node][:pre_upper]
    #end
    # println("=== in relu ===")
    # println("lower: ", lower)
    # println("upper: ", upper)

    alpha_lower = batch_info[node][:alpha_lower]
    alpha_upper = batch_info[node][:alpha_upper]
    upper_slope, upper_bias = relu_upper_bound(lower, upper) #upper_slope:upper of slope  upper_bias:Upper of bias
    lower_bias = prop_method.use_gpu ? fmap(cu, zeros(size(upper_bias))) : zeros(size(upper_bias))

    active_mask = (lower .>= 0)
    inactive_mask = (upper .<= 0)
    unstable_mask = (upper .> 0) .& (lower .< 0)
    batch_info[node][:unstable_mask] = unstable_mask

    beta_lower = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_lower]) : batch_info[node][:beta_lower]
    beta_upper = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_upper]) : batch_info[node][:beta_upper]
    beta_lower_index = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_lower_index]) : batch_info[node][:beta_lower_index]
    beta_upper_index = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_upper_index]) : batch_info[node][:beta_upper_index]
    beta_lower_S = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_lower_S]) : batch_info[node][:beta_lower_S]
    beta_upper_S = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_upper_S]) : batch_info[node][:beta_upper_S]

    lower_A = bound.lower_A_x
    upper_A = bound.upper_A_x
    # println("before lower_A: ")
    # print_beta_layers(lower_A, batch_info[:init_A_b])
    # println("before upper_A: ")
    # print_beta_layers(upper_A, batch_info[:init_A_b])

    batch_info[node][:pre_upper_A_function] = nothing
    batch_info[node][:pre_lower_A_function] = nothing

    if prop_method.bound_lower
        batch_info[node][:pre_lower_A_function] = copy(lower_A)
        Beta_Lower_Layer = BetaLayer(node, alpha_lower, beta_lower, beta_lower_S, beta_lower_index, batch_info[:spec_A_b], true, unstable_mask, active_mask, upper_slope, lower_bias, upper_bias)
        # println("Beta_Lower_Layer.beta_lower: ", Beta_Lower_Layer.beta)
        push!(lower_A, Beta_Lower_Layer)
    end

    if prop_method.bound_upper
        batch_info[node][:pre_upper_A_function] = copy(upper_A)
        Beta_Upper_Layer = BetaLayer(node, alpha_upper, beta_upper, beta_upper_S, beta_upper_index, batch_info[:spec_A_b], false, unstable_mask, active_mask, upper_slope, lower_bias, upper_bias)
        # println("Beta_Upper_Layer.beta_lower: ", Beta_Upper_Layer.beta)
        push!(upper_A, Beta_Upper_Layer)
    end
    push!(batch_info[:Beta_Lower_Layer_node], node)
    New_bound = BetaCrownBound(lower_A, upper_A, nothing, nothing, bound.batch_data_min, bound.batch_data_max)

    # println("after lower_A: ")
    # print_beta_layers(lower_A, batch_info[:init_A_b])
    # println("after upper_A: ")
    # print_beta_layers(upper_A, batch_info[:init_A_b])

    return New_bound
end
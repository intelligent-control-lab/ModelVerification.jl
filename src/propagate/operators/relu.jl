
function propagate_act(prop_method::Union{Ai2z, ImageStarZono}, layer::typeof(relu), reach::AbstractPolytope, batch_info)
    reach = overapproximate(Rectification(reach), Zonotope)
    return reach
end  

function propagate_act(prop_method::Ai2h, layer::typeof(relu), reach::AbstractPolytope, batch_info)
    reach = convex_hull(UnionSetArray(forward_partition(layer, reach)))
    return reach
end

function propagate_act(prop_method::Box, layer::typeof(relu), reach::AbstractPolytope, batch_info)
    reach = rectify(reach)
    return reach
end  

function propagate_act(prop_method, layer::typeof(relu), bound::ImageZonoBound, batch_info)
    cen = reshape(bound.center, :)
    gen = reshape(bound.generators, :, size(bound.generators,4))
    flat_reach = overapproximate(Rectification(Zonotope(cen, gen)), Zonotope)
    new_cen = reshape(LazySets.center(flat_reach), size(bound.center))
    sz = size(bound.generators)
    # println("before size: ", sz)
    new_gen = reshape(genmat(flat_reach), sz[1], sz[2], sz[3], :)
    # println("after size: ", size(new_gen))
    new_bound = ImageZonoBound(new_cen, new_gen)
    return new_bound
end

function propagate_act(prop_method, layer::typeof(relu), bound::Star, batch_info)
    cen = LazySets.center(bound) # h * w * c * 1
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
    return new_bound
end  

function ImageStar_to_Star(bound::ImageStarBound)
    cen = reshape(bound.center, :) # h * w * c * 1
    gen = reshape(bound.generators, :, size(bound.generators,4)) # h*w*c x n_alpha
    T = eltype(cen)
    return Star(T.(cen), T.(gen), HPolyhedron(T.(bound.A), T.(bound.b)))
end

function Star_to_ImageStar(bound::Star, sz)
    new_cen = reshape(LazySets.center(bound), sz[1], sz[2], sz[3], 1)
    new_gen = reshape(basis(bound), sz[1], sz[2], sz[3], :) # h x w x c x (n_alpha + n_beta)
    A = permutedims(cat([con.a for con in constraints_list(bound.P)]..., dims=2)) # n_con x n_alpha
    b = vcat([con.b for con in constraints_list(bound.P)]...) # n_con
    T = eltype(new_cen)
    return ImageStarBound(T.(new_cen), T.(new_gen), T.(A), T.(b))
end

function propagate_act(prop_method, layer::typeof(relu), bound::ImageStarBound, batch_info)
    sz = size(bound.generators)
    flat_bound = ImageStar_to_Star(bound)
    new_flat_bound = propagate_act(prop_method, layer, flat_bound, batch_info)
    new_bound = Star_to_ImageStar(new_flat_bound, sz)
    return new_bound
end

function propagate_act_batch(prop_method::ForwardProp, layer::typeof(relu), bound::CrownBound, batch_info)
    
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
    return new_bound
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




#initalize relu's alpha_lower and alpha_upper
function init_alpha(layer::typeof(relu), node, batch_info)
    relu_input_lower, relu_input_upper = compute_bound(batch[node]["bound"]) # reach_dim x batch #now the bound is crown
    #batch_size = size(relu_input_lower)[end]
    unstable_mask = (relu_input_upper .> 0) .& (relu_input_lower .< 0) #indices of non-zero alphas/ indices of activative neurons
    alpha_indices = findall(unstable_mask) 
    upper_slope, upper_bias = relu_upper_bound(relu_input_lower, relu_input_upper) #upper slope and upper bias
    lower_d = convert(typeof(upper_slope), upper_slope .> 0.5) #lower slope
    push!(batch_info[node], "alpha_shape" => size(lower_d))
    #minimum_sparsity = batch_info[node]["minimum_sparsity"]
    #total_neuron_size = length(relu_input_lower) ÷ batch_size #number of the neuron of the pre_layer of relu

    #fully alpha
    @assert ndims(relu_input_lower) == 2 || ndims(relu_input_lower) == 4 "pre_layer of relu should be dense or conv"
    #if(ndims(relu_input_lower) == 2) #pre_layer of relu is dense 
    #end
    #alpha_lower is for lower bound, alpha_upper is for upper bound
    alpha_lower = alpha_upper = lower_d .* unstable_mask
    push!(batch_info[node], "alpha_lower" => alpha_lower) #reach_dim x batch
    push!(batch_info[node], "alpha_upper" => alpha_upper) #reach_dim x batch
end    



function get_lower_d(lower, upper, lb_lower_slope, ub_lower_slope)
    lower_mask = (lower .>= 0)
    upper_mask = (upper .<= 0)
    unstable_mask = (upper .> 0) .& (lower .< 0)
    
    if !isnothing(lb_lower_slope)
        lb_lower_slope = clamp.(lb_lower_slope, 0.0, 1.0) .* unstable_mask .+ lower_mask #the slope of unstable neuron is alpha, the slope of activative neuron is 1
    end
    
    if !isnothing(ub_lower_slope)
        ub_lower_slope = clamp.(ub_lower_slope, 0.0, 1.0) .* unstable_mask .+ lower_mask #the slope of unstable neuron is alpha, the slope of activative neuron is 1
    end
    
    return lb_lower_slope, ub_lower_slope
end



#Upper bound slope and intercept according to CROWN relaxation.
function relu_upper_bound(lower, upper)
    lower_r = clamp.(lower, -Inf, 0)
    upper_r = clamp.(upper, 0, Inf)
    #lower_r .= min.(lower_r, 0)
    upper_r .= max.(upper_r, lower_r .+ 1e-8)
    upper_slope = upper_r ./ (upper_r .- lower_r) #the slope of the relu upper bound
    upper_bias = - lower_r .* upper_slope #the bias of the relu upper bound
    return upper_slope, upper_bias
end



function backward_relaxation(bound_lower, bound_upper, node, batch_info)
    if !isnothing(batch_info[node]["inputs"])
        input_node = batch_info[node]["inputs"][1]
        lower, upper = compute_bound(batch_info[input_node]["bound"])
    else
        lower, upper = compute_bound(batch_info[node]["bound"])
    end
    lower_bias = nothing
    alpha_lower = batch_info[node]["alpha_lower"]
    alpha_upper = batch_info[node]["alpha_upper"]
    upper_slope, upper_bias = relu_upper_bound(lower, upper) #upper_slope:upper of slope  upper_bias:Upper of bias
    lb_lower_slope, ub_lower_slope = get_lower_d(lower, upper, alpha_lower, alpha_upper) #lower_d：lower of slope lower_d：lower of bias
    return  upper_slope, upper_bias, lb_lower_slope, ub_lower_slope, lower_bias
end 

#bound oneside of the relu, like upper or lower
function bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg)
    if isnothing(last_A)
        return None, 0
    end

    New_A, New_bias = multiply_by_A_signs(last_A, d_pos, d_neg, b_pos, b_neg)
    return New_A, New_bias
end


#using last_A for getting New_A
function multiply_by_A_signs(last_A, d_pos, d_neg, b_pos, b_neg)
    if ndims(d_pos) == 1
        # Special case for LSTM, the bias term is 1-dimension. 
        New_A = clamp.(last_A, 0, Inf) .* d_pos .+ clamp.(last_A, -Inf, 0) .* d_neg
        New_bias = clamp.(last_A, 0, Inf) .* b_pos .+ clamp.(last_A, -Inf, 0) .* b_neg
        return New_A, New_bias
    else
        New_A, New_bias = clamp_mutiply_forward(last_A, d_pos, d_neg, b_pos, b_neg)
        return New_A, New_bias
    end
end


function clamp_mutiply_forward(last_A, d_pos, d_neg, b_pos, b_neg) 
    A_pos = clamp.(last_A, 0, Inf)
    A_neg = clamp.(last_A, -Inf, 0)
    New_A = d_pos .* A_pos .+ d_neg .* A_neg
    bias_pos = bias_neg = [0.0]
    if b_pos !== nothing #bias_pos = torch.einsum('...sb,...sb->sb', A_pos, b_pos)
        s_pos = max(size(A_pos)[end], size(b_pos)[end])
        h_pos = max(size(A_pos)[end-1], size(b_pos)[end-1])
        shape_A_pos = collect(size(A_pos))
        shape_b_pos = collect(size(b_pos))

        shape_A_pos[end] = s_pos 
        shape_A_pos[end-1] = h_pos 
        shape_b_pos[end] = s_pos 
        shape_b_pos[end-1] = h_pos 

        A_pos_repeat_times = shape_A_pos .÷ collect(size(A_pos))
        b_pos_repeat_times = shape_b_pos .÷ collect(size(b_pos))

        bias_pos = zeros(h_pos, s_pos)
        A_pos = repeat(A_pos, outer = A_pos_repeat_times) 
        b_pos = repeat(b_pos, outer = b_pos_repeat_times)
        for i in 1:s_pos
            for j in 1:h_pos
                bias_pos[j, i] = sum(A_pos[:, j, i] .* b_pos[:, j, i])
            end
        end
    end

    if b_neg !== nothing #bias_neg = torch.einsum('...sb,...sb->sb', A_neg, b_neg)
        s_neg = max(size(A_neg)[end], size(b_neg)[end])
        h_neg = max(size(A_neg)[end-1], size(b_neg)[end-1])
        shape_A_neg = collect(size(A_neg))
        shape_b_neg = collect(size(b_neg))

        shape_A_neg[end] = s_neg 
        shape_A_neg[end-1] = h_neg 
        shape_b_neg[end] = s_neg 
        shape_b_neg[end-1] = h_neg 

        A_neg_repeat_times = shape_A_neg .÷ collect(size(A_neg))
        b_neg_repeat_times = shape_b_neg .÷ collect(size(b_neg))

        bias_neg = zeros(h_neg, s_neg)
        A_neg = repeat(A_neg, outer = A_neg_repeat_times) 
        b_neg = repeat(b_neg, outer = b_neg_repeat_times)
        for i in 1:s_neg
            for j in 1:h_neg
                bias_neg[j, i] = sum(A_neg[:, j, i] .* b_neg[:, j, i])
            end
        end
    end
    New_bias = bias_pos .+ bias_neg
    return New_A, New_bias
end 


function propagate_act_batch(prop_method::BackwardProp, layer::typeof(relu), node, bound::AlphaCrownBound, batch_info)
    upper_slope, upper_bias, ub_lower_slope, lb_lower_slope, lower_bias = backward_relaxation(prop_method.bound_lower, prop_method.bound_upper, node, boundbatch_info)

    if prop_method.bound_upper
        uA, ubias = bound_oneside(bound.lower_A_x, upper_slope, ub_lower_slope, upper_bias, lower_bias)
    end
    if prop_method.bound_lower
        lA, lbias = bound_oneside(bound.upper_A_x, lb_lower_slope, upper_slope, lower_bias, upper_bias)
    end
    bound = AlphaCrownBound(lA, uA, nothing, nothing, lbias, ubias, bound.batch_data_min, bound.batch_data_max)
    return bound
end
               
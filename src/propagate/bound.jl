
abstract type Bound end

function init_batch_bound(prop_method::PropMethod, batch_input)
    return [init_bound(prop_method, input) for input in batch_input]
end

function init_bound(prop_method::ForwardProp, input)
    return input
end

function init_bound(prop_method::BackwardProp, input)
    return input
end

struct ImageStarBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  h x w x c x n_gen
    A::AbstractArray{T, 2}            # n_con x n_gen
    b::AbstractArray{T, 1}            # n_con 
end

struct ImageZonoBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  h x w x c x n_gen
end

"""
init_bound(prop_method::ImageStar, batch_input) 

Assume batch_input[1] is a list of vertex images.
Return a zonotope. 

Outputs:
- `ImageStarBound`
"""

function init_bound(prop_method::ImageStar, ch::ImageConvexHull) 
    imgs = ch.imgs
    T = typeof(imgs[1][1,1,1])
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    n = length(imgs)-1 # number of generators
    I = Matrix{T}(LinearAlgebra.I(n))
    A = [I; .-I]
    b = [ones(T, n); ones(T, n)] # -1 to 1
    return ImageStarBound(T.(cen), T.(gen), A, b)
end

function init_bound(prop_method::ImageStarZono, ch::ImageConvexHull) 
    imgs = ch.imgs
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    return ImageZonoBound(cen, gen)
end

function init_bound(prop_method::StarSet, input::Hyperrectangle) 
    isa(input, Star) && return input
    cen = LazySets.center(input) 
    gen = LazySets.genmat(input)
    T = eltype(input)
    n = dim(input)
    I = Matrix{T}(LinearAlgebra.I(n))
    A = [I; .-I]
    b = [ones(T, n); ones(T, n)] # -1 to 1
    return Star(T.(cen), T.(gen), HPolyhedron(A, b))
end


struct LinearBound{T<:Real, F<:AbstractPolytope} <: Bound
    Low::AbstractArray{T, 3} # reach_dim x input_dim x batch_size
    Up::AbstractArray{T, 3}  # reach_dim x input_dim x batch_size
    domain::AbstractArray{F}  
end

struct CrownBound{T<:Real} <: Bound
    batch_Low::AbstractArray{T, 3}    # reach_dim x input_dim+1 x batch_size
    batch_Up::AbstractArray{T, 3}     # reach_dim x input_dim+1 x batch_size
    batch_data_min::AbstractArray{T, 2}     # input_dim+1 x batch_size
    batch_data_max::AbstractArray{T, 2}     # input_dim+1 x batch_size
end
  
struct AlphaCrownBound <: Bound
    lower_A_x::Function
    upper_A_x::Function
    lower_A_W
    upper_A_W
    lower_bias::Function
    upper_bias::Function
    batch_data_min
    batch_data_max
end

function init_batch_bound(prop_method::Crown, batch_input::AbstractArray)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    n = dim(batch_input[1])
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    batch_Low = repeat([I Z], outer=(1, 1, batch_size))
    batch_Up = repeat([I Z], outer=(1, 1, batch_size))
    batch_data_min = cat([low(h) for h in batch_input]..., dims=2)
    batch_data_min = [batch_data_min; ones(batch_size)'] # the last dimension is for bias
    batch_data_max = cat([high(h) for h in batch_input]..., dims=2)
    batch_data_max = [batch_data_max; ones(batch_size)'] # the last dimension is for bias
    bound = CrownBound(batch_Low, batch_Up, batch_data_min, batch_data_max)
    return bound
end

function init_batch_bound(prop_method::AlphaCrown, batch_input::AbstractArray)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    n = dim(batch_input[1])
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    #A = repeat(I, outer=(1, 1, batch_size))
    #b = repeat(Z, outer=(1, 1, batch_size))
    A(x) = repeat(I, outer=(1, 1, batch_size))
    b(x) = repeat(Z, outer=(1, 1, batch_size))
    batch_data_min = cat([low(h) for h in batch_input]..., dims=2)
    batch_data_max = cat([high(h) for h in batch_input]..., dims=2)
    bound = AlphaCrownBound(A, A, nothing, nothing, b, b, batch_data_min, batch_data_max)
    return bound
end

"""   
compute_bound(low::AbstractVecOrMat, up::AbstractVecOrMat, data_min_batch, data_max_batch) where N

Compute lower and upper bounds of a relu node in Crown.
`l, u := ([low]₊*data_min + [low]₋*data_max), ([up]₊*data_max + [up]₋*data_min)`

Outputs:
- `(lbound, ubound)`
"""
function compute_bound(bound::CrownBound)
    # low::AbstractVecOrMat{N}, up::AbstractVecOrMat{N}, data_min_batch, data_max_batch
    # low : reach_dim x input_dim x batch
    # data_min_batch: input_dim x batch
    # l: reach_dim x batch
    # batched_vec is a mutant of batched_mul that accepts batched vector as input.
    z = zeros(size(bound.batch_Low))
    l =   batched_vec(max.(bound.batch_Low, z), bound.batch_data_min) + batched_vec(min.(bound.batch_Low, z), bound.batch_data_max)
    u =   batched_vec(max.(bound.batch_Up, z), bound.batch_data_max) + batched_vec(min.(bound.batch_Up, z), bound.batch_data_min)
    # @assert all(l.<=u) "lower bound larger than upper bound"
    return l, u
end

function compute_bound(bound::AlphaCrownBound)
    z = zeros(size(bound.lower_A_x))
    l = batched_mul(max.(bound.lower_A_x, z), bound.batch_data_min) .+ batched_mul(min.(bound.lower_A_x, z), bound.batch_data_max) .+ bound.lower_bias
    u = batched_mul(max.(bound.upper_A_x, z), bound.batch_data_max) .+ batched_mul(min.(bound.upper_A_x, z), bound.batch_data_min) .+ bound.upper_bias
    return l, u
end


function process_bound(prop_method, batch_bound, batch_out_spec)
    lower_output, upper_output = compute_bound(batch_bound)
    spec_lower_output = batched_mul(batch_out_spec.A, lower_output) .- batch_out_spec.b
    spec_upper_output = batched_mul(batch_out_spec.A, upper_output) .- batch_out_spec.b
    lower_loss = sum(spec_lower_output)
    upper_loss = sum(spec_upper_output)
    optimizer = Flux.Optimiser(Flux.ADAM(0.1))
    for i in 1:prop_method.max_optimize_iter
        Flux.train!(lower_loss, optimizer)
        Flux.train!(upper_loss, optimizer)
    end
end

#= function init_slope()
    for node in Model
        if method in ["backward", "forward+backward"]
            c = share_slopes = final_node_name = nothing
            start_nodes = [start_nodes; get_alpha_crown_start_nodes(
                node, c, share_slopes, final_node_name)]
        end
        init_opt_parameters(start_nodes)
        init_intermediate_bounds[node.inputs[1].name] = (
            [detach(node.inputs[1].lower), detach(node.inputs[1].upper)])
    end
end


function get_alpha_crown_start_nodes(node, c = nothing,  share_slopes = false, final_node_name = nothing)
    sparse_intermediate_bounds = true
    use_full_conv_alpha_thresh = 512
    start_nodes = []
    for nj in backward_from[node]
        unstable_idx = nothing
        use_sparse_conv = nothing
        use_full_conv_alpha = true
        if nj.name == final_node_name
            size_final = isnothing(c) ? final_node_name.output_shape[end] : size(c, 2)
            push!(start_nodes, (final_node_name, size_final, nothing))
            continue
        end
    end 
end =#


#= function compute_bound(x, C, bound::CrownBound, bound_lower = true, bound_upper = false,
    intermediate_layer_bounds, batch_info, Model) # x is the input of the model, C is the constrains of the model
    batch_size = size(Model["model_inputs"])[end]
    dim_in = 0

    #This "for" loop maybe useless
    for node in Model["start_nodes"]
        value = Model["model_inputs"]
        if haskey(batch_info[node], "perturbation_info") #perturbation_info contains information of perturbation, like eps, norm
            ret_init = init_perturbation(node, value, batch_info[node]["perturbation_info"], batch_info, Model)
            push!(batch_info[node], "interval" => [ret_init.batch_Low, ret_init.batch_Up])
            push!(batch_info[node], "lower" => ret_init.batch_Low)
            push!(batch_info[node], "upper" => ret_init.batch_Up)
            push!(batch_info[node], "bound" => ret_init)
        else
            # This input/parameter does not have perturbation.
            push!(batch_info[node], "interval" => [value, value])
            push!(batch_info[node], "forward_value" => value)
            new_bound = CrownBound(value, value, batch_info[node]["data_min"], batch_info[node]["data_max"])
            push!(batch_info[node], "lower" => value)
            push!(batch_info[node], "upper" => value)
            push!(batch_info[node], "bound" => new_bound)
        end
    end
    
    for node in Model["all_nodes"]
        # Check whether all prior intermediate bounds already exist
        push!(batch_info[node], "prior_checked" => false)
        # check whether weights are perturbed and set nonlinear for some operations
        if isa(batch_info[node]["layer"], Flux.Dense) || isa(batch_info[node]["layer"], Flux.Conv) || isa(batch_info[node]["layer"], Flux.BatchNorm)#if the params of Linear, Conv, Batchnorm need to be perturbed, the Linear, Conv, Batchnorm will be non_linear
            push!(batch_info[node], "nonlinear" => false)
            if haskey(batch_info[node], "weight_ptb") || haskey(batch_info[node], "bias_ptb" )
                push!(batch_info[node], "nonlinear" => true)
            end
        end
    end

    final_node = Model["final_nodes"][1]
    set_used_nodes(final_node, batch_info, Model)
    check_prior_bounds(final_node, batch_info, Model)# Maybe useless
    propagate(::BackwardProp, C, node, unstable_idx, unstable_size, batch_info, Model)
end =#  


function set_used_nodes(final_node, batch_info, Model) #finish verifying
    push!(Model, "last_final_node" => nothing)
    if final_node != Model["last_final_node"]
        push!(Model, "last_final_node" => final_node)
        push!(Model, "used_nodes" => [])
        for node in Model["all_nodes"]
            push!(batch_info[node], "used" => false)
        end
        push!(batch_info[final_node], "used" => true)
        queue = Queue{Any}()
        enqueue!(queue, final_node)
        while !isempty(queue)
            node = dequeue!(queue)
            push!(Model["used_nodes"], node)
            if isnothing(batch_info[node]["inputs"])
                continue
            else
                for input_node in batch_info[node]["inputs"]
                    if !batch_info[input_node]["used"]
                        push!(batch_info[input_node], "used" => true)
                        enqueue!(queue, input_node)
                    end
                end
            end
        end
    end
end


#= function check_prior_bounds(node, batch_info, Model)
    if batch_info[node]["prior_checked"] || !(batch_info[node]["used"] && batch_info[node]["ptb"])
        return
    end
    
    if !isnothing(batch_info[node]["inputs"])
        for input_node in batch_info[node]["inputs"]
            check_prior_bounds(input_node, batch_info, Model)
        end
    end

    if haskey(batch_info[node], "nonlinear") && batch_info[node]["nonlinear"]
        for input_node in batch_info[node]["inputs"]
            compute_intermediate_bounds(input_node, batch_info, Model, true)
        end
    end

    if haskey(batch_info[node], "requires_input_bounds")
        for i in batch_info[node]["requires_input_bounds"]
            compute_intermediate_bounds(batch_info[node]["inputs"][i], batch_info, Model, true)
        end
    end
    push!(batch_info[node], "prior_checked" => true)
end

#Haven't finish
function compute_intermediate_bounds(node, batch_info, Model, prior_checked = false)
    if haskey(batch_info[node], "lower")# && !isnothing(batch_info[node]["lower"])
        return
    end

    if !prior_checked
        check_prior_bounds(node, batch_info, Model)
    end

    if !batch_info[node]["ptb"]
        fv = get_forward_value(node)
        push!(batch_info[node], "interval" => [fv, fv])
        push!(batch_info[node], "lower" => fv)
        push!(batch_info[node], "upper" => fv)
        return
    end
end =#


#= function get_sparse_C(node, Model, sparse_intermediate_bounds = true, ref_intermediate_lb = nothing, 
    ref_intermediate_ub = nothing)
    sparse_conv_intermediate_bounds = Model.sparse_conv_intermediate_bounds 
    minimum_sparsity = Model.minimum_sparsity
    crown_batch_size = Model.crown_batch_size
    dim = prod(node.output_shape[2:end])
    batch_size = Model.batch_size
    
    reduced_dim = false  # Only partial neurons (unstable neurons) are bounded.
    unstable_idx = nothing
    unstable_size = Inf
    newC = nothing

    if node.type == "Dense" || node.type == "MatMul"
        if sparse_intermediate_bounds
            # If we are doing bound refinement and reference bounds are given, we only refine unstable neurons.
            # Also, if we are checking against LP solver we will refine all neurons and do not use this optimization.
            # For each batch element, we find the unstable neurons.
            unstable_idx, unstable_size = get_unstable_locations(Model, ref_intermediate_lb, ref_intermediate_ub)
            if unstable_size == 0
                # Do nothing, no bounds will be computed.
                reduced_dim = true
                unstable_idx = []
            elseif unstable_size > crown_batch_size
                # Create C in batched CROWN
                newC = "OneHot"
                reduced_dim = true
            elseif unstable_size <= minimum_sparsity * dim && unstable_size > 0 && isnothing(alpha_is_sparse) || alpha_is_sparse
                # When we already have sparse alpha for this layer, we always use sparse C. Otherwise we determine it by sparsity.
                # Create an abstract C matrix, the unstable_idx are the non-zero elements in specifications for all batches.
                newC = Constarin([batch_size, unstable_size, node.output_shape[1:end]], unstable_idx)
                reduced_dim = true
            else
                unstable_idx = nothing
                ref_intermediate_lb = nothing
                ref_intermediate_ub = nothing
            end
        end
        if !reduced_dim
            newC = eyeC([batch_size, dim, node.output_shape[1:end]]) #another struct, need to be change
        end
    end
    return newC, reduced_dim, unstable_idx, unstable_size
end



function check_optimized_variable_sparsity(node, Model)
    alpha_sparsity = nothing  # unknown
    for relu in Model.relus
        if hasproperty(relu, :alpha_lookup_idx) && node.name in relu.alpha_lookup_idx
            if !isnothing(relu.alpha_lookup_idx[node.name])
                # This node was created with sparse alpha
                alpha_sparsity = true
            else
                alpha_sparsity = false
            end
            break
        end
    end
    return alpha_sparsity
end



function get_unstable_locations(Model, ref_intermediate_lb, ref_intermediate_ub, conv = false, channel_only = false)
        max_crown_size = Model.max_crown_size
        unstable_masks = (ref_intermediate_lb) .< 0 .& (ref_intermediate_ub .> 0)
        if channel_only
            unstable_locs = sum(unstable_masks, dims = (1, 2, 4)) .> 0
            unstable_idx = findall(unstable_locs)
        else
            if !conv && ndims(unstable_masks) > 2
                unstable_masks = reshape(unstable_masks, size(unstable_masks, 1), :)
                ref_intermediate_lb = reshape(ref_intermediate_lb, size(ref_intermediate_lb, 1), :)
                ref_intermediate_ub = reshape(ref_intermediate_ub, size(ref_intermediate_ub, 1), :)
            end
            unstable_locs = sum(unstable_masks, dims = ndims(unstable_masks)) .> 0
            if conv
                unstable_idx = findall(unstable_locs)
            else
                unstable_idx = findall(unstable_locs)
            end
        end
    
        unstable_size = length(unstable_idx)
        if unstable_size > max_crown_size
            indices_selected = select_unstable_idx(ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size)
            if isa(unstable_idx, Tuple)
                unstable_idx = tuple(u[indices_selected] for u in unstable_idx)
            else
                unstable_idx = unstable_idx[indices_selected]
            end
        end
        unstable_size = length(unstable_idx)
    
        return unstable_idx, unstable_size
end =#


struct GradientBound{F<:AbstractPolytope, N<:Real}
    sym::LinearBound{F} # reach_dim x input_dim x batch_size
    LΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
    UΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
end

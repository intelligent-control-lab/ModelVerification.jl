
abstract type Bound end


function init_bound(prop_method::PropMethod, batch_input)
    return batch_input
end

struct ImageStarBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  of  h x w x 4
    P::HPolyhedron                          # n_con x n_gen+1
end

struct ImageZonoBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  of  h x w x 4
end

mutable struct Constrain
    #= batch_size::Int
    unstable_size::Int
    output_size::Any =#
    shape::Any #shape is [batch_size, unstable_size, output_size]
    unstable_idx::Any
    #C::Any
end 
"""
init_bound(prop_method::ImageStar, batch_input) 

Assume batch_input[1] is a list of vertex images.
Return a zonotope. 

Outputs:
- `ImageStarBound`
"""
function init_bound(prop_method::ImageStar, batch_input) 
    # batch_input = [list of vertex images]
    @assert length(batch_input) == 1 "ImageStarBound only support batch_size = 1"
    imgs = batch_input[1]
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    n = length(imgs)-1 # number of generators
    T = typeof(imgs[1][1,1,1])
    I = Matrix{T}(LinearAlgebra.I(n))
    P = HPolyhedron([I; .-I], [ones(T, n); ones(T, n)]) # -1 to 1
    return ImageStarBound(cen, gen, P)
end

function init_bound(prop_method::ImageStarZono, batch_input) 
    # batch_input = [list of vertex images]
    @assert length(batch_input) == 1 "ImageStarBound only support batch_size = 1"
    imgs = batch_input[1]
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    return ImageZonoBound(cen, gen)
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

function init_bound(prop_method::Crown, batch_input::AbstractArray)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    n = dim(batch_input[1])
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    batch_Low = repeat([I Z], outer=(1, 1, batch_size))
    batch_Up = repeat([I Z], outer=(1,1, batch_size))
    batch_data_min = cat([low(h) for h in batch_input]..., dims=2)
    # println("init bound")
    # println(size(batch_data_min))
    # println(size(zeros(batch_size)))
    batch_data_min = [batch_data_min; ones(batch_size)'] # the last dimension is for bias
    batch_data_max = cat([high(h) for h in batch_input]..., dims=2)
    batch_data_max = [batch_data_max; ones(batch_size)'] # the last dimension is for bias
    bound = CrownBound(batch_Low, batch_Up, batch_data_min, batch_data_max)
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
    # println(size(bound.batch_Low))
    # println(size(bound.batch_data_min))
    # println("compute_bound")
    # println("bound.batch_Low")
    # println(bound.batch_Low)
    # println("bound.batch_Up")
    # println(bound.batch_Up)
    # println("bound.batch_data_min")
    # println(bound.batch_data_min)
    # println("bound.batch_data_max")
    # println(bound.batch_data_max)
    
    l =   batched_vec(max.(bound.batch_Low, z), bound.batch_data_min) + batched_vec(min.(bound.batch_Low, z), bound.batch_data_max)
    u =   batched_vec(max.(bound.batch_Up, z), bound.batch_data_max) + batched_vec(min.(bound.batch_Up, z), bound.batch_data_min)
    # println("compute_bound")
    # println("l")
    # println(l)
    # println("u")
    # println(u)
    # println(l.<=u)
    # @assert all(l.<=u) "lower bound larger than upper bound"
    return l, u
end

function compute_bound(x, C, bound::CrownBound, bound_lower = true, bound_upper = false, final_node_name,
    intermediate_layer_bounds, batch_info, global_info) # x is the input of the model, C is the constrains of the model
    # root constrains all the input node, like inputs, input_params
    root = [batch_info[name] for name in root_name] #batch_info includes each node's input, bonud, and so on 
    batch_size = size(root[1].inputs, 1)  # BoundParams = "/1" is 1.weight
    dim_in = 0
    for i in 1:length(root)
        value = forward(root[i])
        if(root[i].ptb == true)
            ret_init = init_perturbation(root[i].node, root[i].batch_input, root[i].perturbation_info)
            root[i].interval = [ret_init.batch_Low, ret_init.batch_Up, root[i].perturbation_info]
            root[i].lower = ret_init.batch_Low
             root[i].upper = ret_init.batch_Up
        else
            # This input/parameter does not have perturbation.
            root[i].interval = [value, value, nothing]
            root[i].forward_value = root[i].value = value
            root[i].lower = root[i].upper = value
        end
    end
    final = batch_info[final_node_name]
    set_used_nodes(final, batch_info, global_info)
end


function set_used_nodes(final, batch_info, global_info)
    if final.name != global_info.last_final_node #global_info stores some extra info like last_final_node, used_nodes
        global_info.last_final_node = final.name
        global_info.used_nodes = []
        for i in batch_info
            i.used = false
        end
        final.used = true
        queue = Queue{Any}()
        enqueue!(q, final)
        while isempty(queue)
            n = dequeue!(queue)
            push!(global_info.used_nodes, n)
            for n_pre in n.inputs #input of the node
                if !n_pre.used
                    n_pre.used = true
                    push!(queue, n_pre)
                end
            end
        end
    end
end



function get_sparse_C(node, global_info, sparse_intermediate_bounds = true, ref_intermediate_lb = nothing, 
    ref_intermediate_ub = nothing)
    sparse_conv_intermediate_bounds = global_info.sparse_conv_intermediate_bounds 
    minimum_sparsity = global_info.minimum_sparsity
    crown_batch_size = global_info.crown_batch_size
    dim = prod(node.output_shape[2:end])
    batch_size = global_info.batch_size
    
    reduced_dim = false  # Only partial neurons (unstable neurons) are bounded.
    unstable_idx = nothing
    unstable_size = Inf
    newC = nothing

    if node.type == "Dense" || node.type == "MatMul"
        if sparse_intermediate_bounds
            # If we are doing bound refinement and reference bounds are given, we only refine unstable neurons.
            # Also, if we are checking against LP solver we will refine all neurons and do not use this optimization.
            # For each batch element, we find the unstable neurons.
            unstable_idx, unstable_size = get_unstable_locations(global_info, ref_intermediate_lb, ref_intermediate_ub)
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



function check_optimized_variable_sparsity(node, global_info)
    alpha_sparsity = nothing  # unknown
    for relu in global_info.relus
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



function get_unstable_locations(global_info, ref_intermediate_lb, ref_intermediate_ub, conv = false, channel_only = false)
        max_crown_size = global_info.max_crown_size
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
end


struct GradientBound{F<:AbstractPolytope, N<:Real}
    sym::LinearBound{F} # reach_dim x input_dim x batch_size
    LΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
    UΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
end

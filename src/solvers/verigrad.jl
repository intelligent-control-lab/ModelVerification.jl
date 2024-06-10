"""
    VeriGrad <: BatchBackwardProp
"""
mutable struct VeriGrad <: BatchBackwardProp 
    use_alpha::Bool
    use_beta::Bool
    use_gpu::Bool
    pre_bound_method::Union{BatchForwardProp, BatchBackwardProp, Nothing, Dict}
    bound_lower::Bool
    bound_upper::Bool
    optimizer
    train_iteration::Int
    inherit_pre_bound::Bool
    Lie_linear_bound
end
VeriGrad(nothing) = VeriGrad(true, true, true, nothing, true, true, Flux.ADAM(0.1), 10, true, nothing)
VeriGrad(nothing; use_gpu=false) = VeriGrad(true, true, true, nothing, true, true, Flux.ADAM(0.1), 10, true, nothing)
function VeriGrad(use_alpha, use_beta, use_gpu, pre_bound_method, bound_lower, bound_upper, optimizer, train_iteration, inherit_pre_bound)
    return VeriGrad(use_alpha, use_beta, use_gpu, pre_bound_method, bound_lower, bound_upper, optimizer, train_iteration, inherit_pre_bound, nothing)
end
function VeriGrad(;use_alpha=true, use_beta=true, use_gpu=true, pre_bound_method=:BetaCrown, bound_lower=true, bound_upper=true, optimizer=Flux.ADAM(0.1), train_iteration=10, inherit_pre_bound=true, Lie_linear_bound=nothing)
    if pre_bound_method == :BetaCrown
        # pre_bound_method method must inherit_pre_bound, otherwise bound of previous layer will not be memorized in pre-bounding.
        pre_bound_method = VeriGrad(use_alpha, use_beta, use_gpu, nothing, bound_lower, bound_upper, optimizer, train_iteration, true, nothing)
    end
    VeriGrad(use_alpha, use_beta, use_gpu, pre_bound_method, bound_lower, bound_upper, optimizer, train_iteration, inherit_pre_bound,Lie_linear_bound)
end


"""
    VeriGradBound <: Bound
"""
mutable struct VeriGradBound <: Bound
    batch_Low    # reach_dim x input_dim+1 x batch_size
    batch_Up     # reach_dim x input_dim+1 x batch_size
    batch_data_min    # input_dim+1 x batch_size
    batch_data_max     # input_dim+1 x batch_size
    img_size    # width x height x channel or nothing if the input is not ImageConvexHull
end
function VeriGradBound(batch_Low, batch_Up, batch_data_min, batch_data_max)
    return VeriGradBound(batch_Low, batch_Up, batch_data_min, batch_data_max, nothing)
end
function VeriGradBound(crown_bound::CrownBound)
    return VeriGradBound(crown_bound.batch_Low, crown_bound.batch_Up, crown_bound.batch_data_min, crown_bound.batch_data_max, crown_bound.img_size)
end

"""
    ConcretizeVeriGradBound <: Bound
"""
struct ConcretizeVeriGradBound <: Bound
    spec_l
    spec_u
    # batch_data_min
    # batch_data_max
end

# function compute_bound(bound::VeriGradBound)
#     @assert false
#     compute_bound = Compute_bound(bound.batch_data_min, bound.batch_data_max)
#     bound_lower_model = Chain(push!(bound.lower_A_x, compute_bound)) 
#     bound_upper_model = Chain(push!(bound.upper_A_x, compute_bound)) 
#     use_gpu = any(param -> param isa CuArray, Flux.params(bound.upper_A_x))
#     bound_lower_model = use_gpu ? bound_lower_model |> gpu : bound_lower_model
#     bound_upper_model = use_gpu ? bound_upper_model |> gpu : bound_upper_model
    
#     # @show batch_size
#     # @show n
#     # @show size(bound.batch_data_min)
#     # @show size(bound.lower_A_x[1])

#     batch_size = size(bound.batch_data_min)[end]
#     n = size(bound.lower_A_x[1])[end-1]

#     bound_A_b = init_A_b(n, batch_size)

#     if length(Flux.params(bound_lower_model)) > 0
#         loss_func = x -> -sum(x[1].^2) # surrogate loss to maximize the min
#         @timeit to "optimize_model" bound_lower_model = optimize_model(bound_lower_model, bound_A_b, loss_func, prop_method.optimizer, prop_method.train_iteration)
#     end
#     if length(Flux.params(bound_upper_model)) > 0
#         loss_func = x -> sum(x[2].^2) # surrogate loss to minimize the max
#         @timeit to "optimize_model" bound_upper_model = optimize_model(bound_upper_model, bound_A_b, loss_func, prop_method.optimizer, prop_method.train_iteration)
#     end
#     lower_l, lower_u = bound_lower_model(bound_A_b)
#     upper_l, upper_u = bound_upper_model(bound_A_b)
#     return lower_l, upper_u
# end 

# """
#     Compute_verigrad_bound
# """
# struct Compute_verigrad_bound
#     batch_data_min
#     batch_data_max
# end
# Flux.@functor Compute_verigrad_bound ()


# function (f::Compute_verigrad_bound)(x)
#     A_pos = clamp.(x[1], 0, Inf)
#     A_neg = clamp.(x[1], -Inf, 0)
#     # @show size(f.batch_data_min), size(A_pos)
#     l = batched_vec(A_pos, f.batch_data_min) + batched_vec(A_neg, f.batch_data_max) .+ x[2]
#     u = batched_vec(A_pos, f.batch_data_max) + batched_vec(A_neg, f.batch_data_min) .+ x[2]
#     return l, u
# end 

"""   
    compute_bound(bound::VeriGradBound)

Compute lower and upper bounds of a relu node in Crown.
`l, u := ([low]₊*data_min + [low]₋*data_max), ([up]₊*data_max + [up]₋*data_min)`

## Arguments
- `bound` (`VeriGradBound`): VeriGradBound object
Outputs:
- `(lbound, ubound)`
"""
function compute_bound(bound::VeriGradBound)
    # low::AbstractVecOrMat{N}, up::AbstractVecOrMat{N}, data_min_batch, data_max_batch
    # low : reach_dim x input_dim x batch
    # data_min_batch: input_dim x batch
    # l: reach_dim x batch
    # batched_vec is a mutant of batched_mul that accepts batched vector as input.
    #z =   fmap(cu, zeros(size(bound.batch_Low)))
    #l =   batched_vec(max.(bound.batch_Low, z), bound.batch_data_min) + batched_vec(min.(bound.batch_Low, z), bound.batch_data_max)
    #u =   batched_vec(max.(bound.batch_Up, z), bound.batch_data_max) + batched_vec(min.(bound.batch_Up, z), bound.batch_data_min)
    if length(size(bound.batch_Low)) > 3
        img_size = size(bound.batch_Low)[1:3]
        batch_Low= reshape(bound.batch_Low, (img_size[1]*img_size[2]*img_size[3], size(bound.batch_Low)[4],size(bound.batch_Low)[5]))
        batch_Up= reshape(bound.batch_Up, (img_size[1]*img_size[2]*img_size[3], size(bound.batch_Up)[4],size(bound.batch_Up)[5]))
        l =   batched_vec(clamp.(batch_Low, 0, Inf), bound.batch_data_min) .+ batched_vec(clamp.(batch_Low, -Inf, 0), bound.batch_data_max)
        u =   batched_vec(clamp.(batch_Up, 0, Inf), bound.batch_data_max) .+ batched_vec(clamp.(batch_Up, -Inf, 0), bound.batch_data_min)
    else
        l =   batched_vec(clamp.(bound.batch_Low, 0, Inf), bound.batch_data_min) .+ batched_vec(clamp.(bound.batch_Low, -Inf, 0), bound.batch_data_max)
        u =   batched_vec(clamp.(bound.batch_Up, 0, Inf), bound.batch_data_max) .+ batched_vec(clamp.(bound.batch_Up, -Inf, 0), bound.batch_data_min)
    end
    # @assert all(l.<=u) "lower bound larger than upper bound"
    return l, u
end



"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::VeriGrad, problem::Problem)
"""
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::VeriGrad, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    model = prop_method.use_gpu ? problem.Flux_model |> gpu : problem.Flux_model
    return model_info, Problem(problem.onnx_model_path, model, init_bound(prop_method, problem.input), problem.output)
end

"""
    init_batch_bound(prop_method::VeriGrad, batch_input::AbstractArray, out_specs)
"""
function init_batch_bound(prop_method::VeriGrad, batch_input::AbstractArray, out_specs)
    # initial bound for the original NN
    # @show size(out_specs.A)
    batch_size = length(batch_input)
    img_size = nothing
    if typeof(batch_input[1]) == ReLUConstrainedDomain
        batch_input = [b.domain for b in batch_input]
    end
    if typeof(batch_input[1]) == ImageConvexHull
        # convert batch_input from list of ImageConvexHull to list of Hyperrectangle
        img_size = ModelVerification.get_size(batch_input[1])
        @assert all(<=(0), batch_input[1].imgs[1]-batch_input[1].imgs[2]) "the first ImageConvexHull input must be upper bounded by the second ImageConvexHull input"
        batch_input = [Hyperrectangle(low = reduce(vcat,img_CH.imgs[1]), high = reduce(vcat,img_CH.imgs[2]))  for img_CH in batch_input]
    end
    m = 1
    n = prop_method.use_gpu ? fmap(cu, dim(batch_input[1])) : dim(batch_input[1])
    @assert size(out_specs.A)[2] == n "the gradient should be with the same size of input, currently Jacobian is not supported"
    I = prop_method.use_gpu ? fmap(cu, Matrix{Float64}(LinearAlgebra.zeros(m,n))) : Matrix{Float64}(LinearAlgebra.zeros(m,n))
    Z = prop_method.use_gpu ? fmap(cu, ones(m)) : ones(m)
    batch_Low = prop_method.use_gpu ? fmap(cu, repeat([I Z], outer=(1, 1, batch_size))) : repeat([I Z], outer=(1, 1, batch_size))
    batch_Up = prop_method.use_gpu ? fmap(cu, repeat([I Z], outer=(1, 1, batch_size))) : repeat([I Z], outer=(1, 1, batch_size))
    batch_data_min = prop_method.use_gpu ? fmap(cu, cat([low(h) for h in batch_input]..., dims=2)) : cat([low(h) for h in batch_input]..., dims=2)
    batch_data_min = prop_method.use_gpu ? fmap(cu, [batch_data_min; ones(batch_size)']) : [batch_data_min; ones(batch_size)']# the last dimension is for bias
    batch_data_max = prop_method.use_gpu ? fmap(cu, cat([high(h) for h in batch_input]..., dims=2)) : cat([high(h) for h in batch_input]..., dims=2)
    batch_data_max = prop_method.use_gpu ? fmap(cu, [batch_data_max; ones(batch_size)']) : [batch_data_max; ones(batch_size)']# the last dimension is for bias
    if !isnothing(img_size) 
        # restore the image size for lower and upper bounds
        batch_Low = reshape(batch_Low, (img_size..., size(batch_Low)[2],size(batch_Low)[3]))
        batch_Up = reshape(batch_Up, (img_size..., size(batch_Up)[2],size(batch_Up)[3]))
    end
    # @show size(batch_data_min)
    bound = VeriGradBound(batch_Low, batch_Up, batch_data_min, batch_data_max, img_size)
    # @show bound
    # @show size(reshape(batch_Low, (img_size..., size(batch_Low)[2],size(batch_Low)[3])))
    return bound
end

"""
    prepare_method(prop_method::VeriGrad, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info)
"""
function prepare_method(prop_method::VeriGrad, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info, sub=false)
    # println("start prepare method")
    out_specs = get_linear_spec(batch_output)
    
    if prop_method.use_gpu
        out_specs = LinearSpec(fmap(cu, out_specs.A), fmap(cu, out_specs.b), fmap(cu, out_specs.is_complement))
    end
    return prepare_method(prop_method, batch_input, out_specs, batch_inheritance, model_info,sub)
end

# # merge a list of inheritance info into a batch
# function batchify_inheritance(inheritance_list::AbstractVector, node, key, use_gpu)
#     eltype(inheritance_list) == Nothing && return nothing
#     n_dim = ndims(inheritance_list[1][node][key])
#     # @show n_dim, node, size(inheritance_list[1][node][key])
#     batch_value = cat([ih[node][key] for ih in inheritance_list]..., dims=n_dim)
#     # @show size(batch_value)
#     batch_value = use_gpu ? batch_value |> gpu : batch_value
#     return batch_value
# end


"""
    prepare_method(prop_method::VeriGrad, batch_input::AbstractVector, out_specs::LinearSpec, model_info)
"""
function prepare_method(prop_method::VeriGrad, batch_input::AbstractVector, out_specs::LinearSpec, inheritance_list::AbstractVector, model_info, sub=false)
    # println("start prepare method, out_specs is already linear")
    # initialize the bound of final node, but it should be updated using the pre-activation bound
    # @show batch_input
    batch_info = init_propagation(prop_method, batch_input, out_specs, model_info)
    
    
    f_node = model_info.final_nodes[1]
    # @show batch_info[f_node][:bound]
    init_size = isnothing(batch_info[f_node][:bound].img_size) ? size(batch_info[f_node][:bound].batch_data_min)[1]-1 : batch_info[f_node][:bound].img_size
    # @show model_info
    # @show batch_info
    # @show init_size
    batch_info = get_all_layer_output_size(model_info, batch_info, init_size)
    # @show batch_info[f_node][:size_after_layer][1]
    @assert batch_info[f_node][:size_after_layer][1] == 1 "currently only support 1-dim-output gradient, not general Jacobian"
    
    # @show batch_info
    batch_info[:spec_A_b] = [out_specs.A, .-out_specs.b] # spec_A x < spec_b  ->  A x + b < 0, need negation

    # println("list_inheritance: ", inheritance_list)

    # println("batch_inheritance: ", batch_inheritance)

    if prop_method.inherit_pre_bound && eltype(inheritance_list) != Nothing # pre_bound can be inherited from the parent branch 
        # println("inheritating pre bound ...")
        for node in model_info.activation_nodes
            # @show node
            # @show batch_inheritance[node]
            # println("batch_inheritance[node][:pre_lower]:", batch_inheritance[node][:pre_lower])
            batch_info[node][:pre_lower] = batchify_inheritance(inheritance_list, node, :pre_lower, prop_method.use_gpu)
            batch_info[node][:pre_upper] = batchify_inheritance(inheritance_list, node, :pre_upper, prop_method.use_gpu)
            batch_info[node][:symbolic_pre_bound] = batchify_inheritance(inheritance_list, node, :symbolic_pre_bound, prop_method.use_gpu)
            if haskey(inheritance_list[1][node], :lower_bound_alpha)
                batch_info[node][:lower_bound_alpha] = batchify_inheritance(inheritance_list, node, :lower_bound_alpha, prop_method.use_gpu)
            end
            if haskey(inheritance_list[1][node], :upper_bound_alpha)
                batch_info[node][:upper_bound_alpha] = batchify_inheritance(inheritance_list, node, :upper_bound_alpha, prop_method.use_gpu)
            end
            # @show node, size(batch_info[node][:pre_lower])
        end
        # println("---done iterating act nodes ---")
    elseif prop_method.pre_bound_method isa BetaCrown  # requires recursive bounding, iterate from first layer

        # println("---computing pre bound ---")
        batch_info = joint_optimization(prop_method.pre_bound_method, batch_input, model_info, batch_info)
        
    elseif !isnothing(prop_method.pre_bound_method) # pre-bounding with other methods
        # println("---computing pre bound ---")
        pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, out_specs, [nothing], model_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            batch_info[prev_node][:symbolic_crown_bound] = pre_batch_info[prev_node][:bound] 
            pre_bound = pre_batch_info[prev_node][:bound]
            batch_info[node][:pre_lower], batch_info[node][:pre_upper] = compute_bound(pre_bound) # reach_dim x batch 
            batch_info[node][:symbolic_pre_bound] = pre_bound
            # @show prev_node, node # node is with relu
            # println("assigning", node," ", prev_node)
        end
        # println("=== Done computing pre bound ===")
    end
    # @show sub
    
    batch_info[:batch_size] = length(batch_input)
    for node in model_info.all_nodes
        batch_info[node][:beta] = 1
        batch_info[node][:max_split_number] = 1
        batch_info[node][:weight_ptb] = false
        batch_info[node][:bias_ptb] = false
    end
    
    
    # init alpha and beta in case they are not initialized in the pre-bounding, or not inheritated.
    for node in model_info.activation_nodes 
        # only init alpha and beta that are not inheritated
        # in the code, we inheritated alpha, but not beta.
        # TODO: modify the inehritance when branching ReLU, then inheritate Beta in filter_and_batchify_inheritance.
        if !haskey(batch_info[node], :lower_bound_alpha) || !haskey(batch_info[node], :lower_bound_alpha)
            batch_info = init_node_alpha(model_info.node_layer[node], node, batch_info, batch_input)
        end
        if !haskey(batch_info[node], :beta_lower) || !haskey(batch_info[node], :beta_upper)
            batch_info = init_beta(model_info.node_layer[node], node, batch_info, batch_input)
        end
        # @show "act", node
        # prev_node = model_info.node_prevs[node][1]
        batch_info[node][:symbolic_pre_bound] = VeriGradBound(batch_info[node][:symbolic_pre_bound])
        # batch_info[node][:bound] = batch_info[node][:symbolic_pre_bound]
        # @show batch_info[node][:bound]
    end
    # batch_info[model_info.final_nodes[1]][:bound] = init_batch_bound(prop_method, model_info, batch_info)
    n = size(out_specs.A, 2)

    batch_info[:init_A_b] = init_A_b(n, batch_info[:batch_size])

    batch_info[:Beta_Lower_Layer_node] = []#store the order of the node which has AlphaBetaLayer
    return out_specs, batch_info

end 



"""
    get_inheritance(prop_method::VeriGrad, batch_info::Dict, batch_idx::Int)

Extract useful informations from batch_info.
These information will later be inheritated by the new branch created by split.

## Arguments
- `prop_method` (`ForwardProp`): Solver being used.
- `batch_info` (`Dict`): all the information collected in propagation.
- `batch_idx`: the index of the interested branch in the batch.
- `model_info`: the general computational graph

## Returns
- `inheritance`: a dict that contains all the information will be inheritated.
"""
function get_inheritance(prop_method::VeriGrad, batch_info::Dict, batch_idx::Int, model_info)
    prop_method.inherit_pre_bound || return nothing
    inheritance = Dict()
    # println("batch_info")
    # println(keys(batch_info))
    for node in model_info.activation_nodes
        # println(size(batch_info[node][:pre_lower]))
        inheritance[node] = Dict(
            :pre_lower => batch_info[node][:pre_lower][:,batch_idx:batch_idx],
            :pre_upper => batch_info[node][:pre_upper][:,batch_idx:batch_idx],
            :symbolic_pre_bound => batch_info[node][:symbolic_pre_bound][:,batch_idx:batch_idx]
        )
    end
    # println("inheritance: ", inheritance)
    return inheritance
end


"""
    update_bound_by_relu_con(node, batch_input, relu_input_lower, relu_input_upper)
"""
function update_bound_by_relu_con(node, batch_input, relu_input_lower, relu_input_upper)
    for input in batch_input
        relu_con_dict = input.all_relu_cons
        if haskey(relu_con_dict,node) && !isnothing(relu_con_dict[node].history_split)
            # println("a")
            # println(a)
            # println("batch_info[node][:pre_lower]")
            # println(batch_info[node][:pre_lower])
            relu_input_lower[relu_con_dict[node].history_split .== 1] .= 0 # enforce relu > 0
            relu_input_upper[relu_con_dict[node].history_split .== -1] .= 0 # enforce relu < 0
        end
    end
    return relu_input_lower, relu_input_upper
end

"""
    init_node_alpha(layer::typeof(relu), node, batch_info, batch_input)
"""
function init_node_alpha(layer::typeof(relu), node, batch_info, batch_input)
    
    # relu_input_lower, relu_input_upper = update_bound_by_relu_con(node, batch_input, relu_input_lower, relu_input_upper)

    l = batch_info[node][:pre_lower]
    u = batch_info[node][:pre_upper]
    # @show ndims(l), node
    
    @assert ndims(l) == 2 || ndims(l) == 4 "pre_layer of relu should be dense or conv"

    unstable_mask = (u .> 0) .& (l .< 0) #indices of non-zero alphas/ indices of activative neurons
    alpha_indices = findall(unstable_mask) 
    upper_slope, upper_bias = relu_upper_bound(l, u) #upper slope and upper bias
    # lower_slope = convert(typeof(upper_slope), upper_slope .> 0.5) #lower slope
    # @show upper_slope
    lower_slope = deepcopy(upper_slope) #lower slope
    # lower_slope = zeros(size(upper_slope))

    lower_slope = isa(l, CuArray) ? lower_slope |> gpu : lower_slope
    
    lower_bound_alpha = lower_slope .* unstable_mask
    upper_bound_alpha = lower_slope .* unstable_mask

    batch_info[node][:lower_bound_alpha] = lower_bound_alpha #reach_dim x batch
    batch_info[node][:upper_bound_alpha] = upper_bound_alpha #reach_dim x batch
    
    return batch_info
end   

#initalize relu's beta

"""
init_beta(layer::typeof(relu), node, batch_info, batch_input)
"""
function init_beta(layer::typeof(relu), node, batch_info, batch_input)

    input_dim = size(batch_info[node][:pre_lower])[1:end-1]
    batch_size = size(batch_info[node][:pre_lower])[end] # TODO: need to be replaced for batched input
    # println("node")
    # println(node)
    # println("input_dim")
    # println(input_dim)
    # println("batch_size")
    # println(batch_size)
    # @assert false
    batch_info[node][:beta_lower] =  zeros(input_dim..., batch_size) # reach_dim x batch 
    batch_info[node][:beta_upper] =  zeros(input_dim..., batch_size)
    batch_info[node][:beta_lower_index] =  []
    batch_info[node][:beta_upper_index] =  []
    batch_info[node][:beta_lower_S] =  zeros(input_dim..., batch_size)
    batch_info[node][:beta_upper_S] =  zeros(input_dim..., batch_size)
    for (i,input) in enumerate(batch_input)
        relu_con_dict = input.all_relu_cons
        if haskey(relu_con_dict,node) && !isnothing(relu_con_dict[node].history_split)
            # println("node")
            # println(node)
            # println(relu_con_dict[node].history_split)
            # sleep(0.1)
            # @assert false
            # println("size(batch_info[node][:beta_lower_S][:,i])")
            # println(size(batch_info[node][:beta_lower_S][:,i]))
            batch_info[node][:beta_lower_S][:,i] = relu_con_dict[node].history_split
            batch_info[node][:beta_upper_S][:,i] = relu_con_dict[node].history_split
        end
    end
    
    for input in batch_input
        relu_con_dict = input.all_relu_cons
        if haskey(relu_con_dict,node)
            push!(batch_info[node][:beta_lower_index], relu_con_dict[node].idx_list)
            push!(batch_info[node][:beta_upper_index], relu_con_dict[node].idx_list)
        end
    end
    
    return batch_info
end


"""
    init_A_b(n, batch_size) # A x < b
"""
function init_A_b(n, batch_size) # A x < b
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    A = repeat(I, outer=(1, 1, batch_size))
    b = repeat(Z, outer=(1, batch_size))
    return [A, b]
end

"""
    init_bound(prop_method::VeriGrad, input) 
"""
function init_bound(prop_method::VeriGrad, input) 
    return ReLUConstrainedDomain(input, Dict())
end
function print_beta_layers(layers, x)
    layers = layers |> gpu
    x = x |> gpu
    println("--- printing beta layers ---")
    println(x)
    for layer in layers
        x = layer(x)
        if isa(layer, BetaLayer)
            println("relu: is_lower ", layer.lower)
            println("u_slope: ", layer.upper_slope)
            lower_slope = clamp.(layer.alpha, 0, 1) .* layer.unstable_mask .+ layer.active_mask 
            println("alpha: ", layer.alpha)
            println("unstable_mask: ", layer.unstable_mask)
            println("l_slope: ", lower_slope)
        else
            println("dense")
        end
        println(x)
    end
    println("--- --- ---")
end

"""
    optimize_model(model, input, loss_func, optimizer, max_iter)
"""
function optimize_model(model, input, loss_func, optimizer, max_iter)
    to = get_timer("Shared")
    
    min_loss = Inf
    @timeit to "setup" opt_state = Flux.setup(optimizer, model)
    for i in 1 : max_iter
        @timeit to "forward" begin
            x = input
            for layer in model
                # println("string(nameof(typeof(layer)))", string(nameof(typeof(layer))))
                @timeit to string(nameof(typeof(layer))) x = layer(x)
                # x = layer(x)
            end
        end
        @timeit to "forward_grad" losses, grads = Flux.withgradient(model) do m
            # println("input")
            # println(input)
            # println("m")
            # println(m)
            result = m(input)
            # println("result: ", result)
            loss_func(result)
        end
        # println("opt_state: ", opt_state)
        # println("losses: ", losses)
        # for p in Flux.params(model)
        #     println("  ",p)
        # end
        # if losses <= min_loss
        #     min_loss = losses
        # else
        #     return model
        # end
        @timeit to "update" Flux.update!(opt_state, model, grads[1])
    end
    return model
end



"""
    get_pre_relu_A(init, use_gpu, lower_or_upper, model_info, batch_info)
"""
function get_pre_relu_A(init, use_gpu, lower_or_upper, model_info, batch_info)
    if lower_or_upper
        for node in model_info.activation_nodes
            A_function = use_gpu ? fmap(cu, Chain(batch_info[node][:pre_lower_A_function])) : Chain(batch_info[node][:pre_lower_A_function])
            batch_info[node][:pre_lower_A] = A_function(init)[1]
            batch_info[node][:pre_upper_A] = nothing
        end
    end
    if !lower_or_upper
        for node in model_info.activation_nodes
            A_function = use_gpu ? fmap(cu, Chain(batch_info[node][:pre_upper_A_function])) : Chain(batch_info[node][:pre_upper_A_function])
            batch_info[node][:pre_upper_A] = A_function(init)[1]
            batch_info[node][:pre_lower_A] = nothing
        end
    end
    return batch_info
end

"""
    get_pre_relu_spec_A(init, use_gpu, lower_or_upper, model_info, batch_info)
"""
function get_pre_relu_spec_A(init, use_gpu, lower_or_upper, model_info, batch_info)
    if lower_or_upper
        for node in model_info.activation_nodes
            A_function = use_gpu ? fmap(cu, Chain(batch_info[node][:pre_lower_A_function])) : Chain(batch_info[node][:pre_lower_A_function])
            batch_info[node][:pre_lower_spec_A] = A_function(init)[1]
            batch_info[node][:pre_upper_spec_A] = nothing
        end
    end
    if !lower_or_upper
        for node in model_info.activation_nodes
            # @show "act ", node
            A_function = use_gpu ? fmap(cu, Chain(batch_info[node][:pre_upper_A_function])) : Chain(batch_info[node][:pre_upper_A_function])
            batch_info[node][:pre_upper_spec_A] = A_function(init)[1]
            batch_info[node][:pre_lower_spec_A] = nothing
        end
    end
    return batch_info
end

"""
    check_inclusion(prop_method::VeriGrad, model, batch_input::AbstractArray, bound::VeriGradBound, batch_out_spec::LinearSpec)
"""
function check_inclusion(prop_method::VeriGrad, model, batch_input::AbstractArray, bound::VeriGradBound, batch_out_spec::LinearSpec)
    # l, u: out_dim x batch_size
    # @show bound.batch_Low[:, end, :]
    if !isnothing(prop_method.Lie_linear_bound)
        # @show l, u = compute_bound(bound)
        bound_crown = Lie_IA_bound(prop_method, batch_input, bound)
        # @show l, u = compute_bound(bound_crown)
        if prop_method.Lie_linear_bound[6] > 0
            # @show model, batch_input
            bias_crown_bound = ModelVerification.find_crown_bound(model, batch_input)
            bound_crown = CrownBound(bound_crown.batch_Low+prop_method.Lie_linear_bound[6] .* bias_crown_bound.batch_Low, bound_crown.batch_Up+prop_method.Lie_linear_bound[6] .* bias_crown_bound.batch_Up,  bound_crown.batch_data_min, bound_crown.batch_data_max, bound_crown.img_size)
        end

        l, u = compute_bound(bound_crown)
        # @show l, u
        # @show size(l)
        batch_size = size(l,2)
        
        # batch_out_spec_A = ones(1, 1, batch_size) # spec_dim x out_dim x batch_size
        # batch_out_spec_b = zeros(1, batch_size) # spec_dim x batch_size

        # pos_A = clamp.(batch_out_spec_A, 0, Inf)
        # neg_A = clamp.(batch_out_spec_A, -Inf, 0)
        # spec_u = batched_vec(pos_A, u) + batched_vec(neg_A, l) .- batch_out_spec_b # spec_dim x batch_size
        # spec_l = batched_vec(pos_A, l) + batched_vec(neg_A, u) .- batch_out_spec_b # spec_dim x batch_size
        # @show spec_l,spec_u
        CUDA.@allowscalar center = (bound_crown.batch_data_min[1:end-1,:] + bound_crown.batch_data_max[1:end-1,:])./2 # out_dim x batch_size

        model = prop_method.use_gpu ? model |> gpu : model

        if !isnothing(bound_crown.img_size)
            center = reshape(center, (bound_crown.img_size..., size(center)[2]))
        end
        # out_center = model(center)
        out_center = jacobian(model, center)[1]'
        if all(out_center .== 0.0)
            out_center = jacobian(model, center .+ tol)[1]'
        end
        # @show out_center
        out_center = out_center' * prop_method.Lie_linear_bound[5]+ prop_method.Lie_linear_bound[6] * model(center)

        # center_res = batched_vec(batch_out_spec_A, out_center) .- batch_out_spec_b # spec_dim x batch_size

        results = []
        spec_u, spec_l = u, l
        center_res = out_center
        # spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
        # spec_l = reshape(maximum(spec_l, dims=1), batch_size) # batch_size, min_x max_i of ai x - bi
        # @show batch_input
        # println("spec")
        # println(spec_l)
        # println(spec_u)
        center_res = reshape(maximum(center_res, dims=1), batch_size) # batch_size
        # if batch_out_spec.is_complement 
        #     # A x < b descript the unsafe set, violated if exist x such that max spec ai x - bi <= 0    
        #     for i in 1:batch_size
        #         CUDA.@allowscalar center_res[i] <= -tol && (results[i] = BasicResult(:violated))
        #         CUDA.@allowscalar spec_l[i] > -tol && (results[i] = BasicResult(:holds))
        #     end
        # else # holds if forall x such that max spec ai x - bi <= tol
        for i in 1:batch_size
            # @show 1e4 * tol
            if CUDA.@allowscalar spec_u[i] <= tol #|| minimum(radius_hyperrectangle(batch_input[i].domain)) <= tol
                push!(results, BasicResult(:holds)) #results[i] = BasicResult(:holds)
            end
            # @show spec_l[i], spec_u[i]
            # CUDA.@allowscalar center_res[i] > tol && (results[i] = CounterExampleResult(:violated, center)) #BasicResult(:violated))
            if CUDA.@allowscalar center_res[i] > tol 
                # @show center
                push!(results, CounterExampleResult(:violated, center))
                # push!(results, BasicResult(:unknown))
                # results[i] = CounterExampleResult(:violated, center)
            else
                push!(results, BasicResult(:unknown))
            end
        end
        # end
    else
        # @show bound
        # @show size(batch_out_spec.A), batch_out_spec.A
        l, u = compute_bound(bound)
        # @show l, u
        # @show minimum(radius_hyperrectangle(batch_input[1].domain))
        
        # @show size(l)
        batch_size = size(l,2)
        #pos_A = max.(batch_out_spec.A, fmap(cu, zeros(size(batch_out_spec.A)))) # spec_dim x out_dim x batch_size
        #neg_A = min.(batch_out_spec.A, fmap(cu, zeros(size(batch_out_spec.A))))
        # @show size(batch_out_spec.A)
        pos_A = clamp.(batch_out_spec.A, 0, Inf)
        neg_A = clamp.(batch_out_spec.A, -Inf, 0)
        spec_u = batched_vec(pos_A, u) + batched_vec(neg_A, l) .- batch_out_spec.b # spec_dim x batch_size
        
        spec_l = batched_vec(pos_A, l) + batched_vec(neg_A, u) .- batch_out_spec.b # spec_dim x batch_size
        # @show spec_l,spec_u
        CUDA.@allowscalar center = (bound.batch_data_min[1:end-1,:] + bound.batch_data_max[1:end-1,:])./2 # out_dim x batch_size
        # println("typeof(center)")
        # println(typeof(center))
        # println("typeof(model)")
        # println(typeof(model))
        model = prop_method.use_gpu ? model |> gpu : model
        # @show model[end-3:end]
        # @show size(center)
        if !isnothing(bound.img_size)
            # resize input to match Conv for image
            # @assert length(size(bound.img_size)) == 3
            center = reshape(center, (bound.img_size..., size(center)[2]))
        end
        # out_center = model(center)
        out_center = jacobian(model, center)[1]'
        if all(out_center .== 0.0)
            out_center = jacobian(model, center .+ tol)[1]'
        end


        center_res = batched_vec(batch_out_spec.A, out_center) .- batch_out_spec.b # spec_dim x batch_size
        # results = [BasicResult(:unknown) for _ in 1:batch_size]
        # results = [CounterExampleResult(:unknown, center) for _ in 1:batch_size]
        results = []
        spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
        spec_l = reshape(maximum(spec_l, dims=1), batch_size) # batch_size, min_x max_i of ai x - bi
        # @show batch_input
        # println("spec")
        # println(spec_l)
        # println(spec_u)
        center_res = reshape(maximum(center_res, dims=1), batch_size) # batch_size
        if batch_out_spec.is_complement 
            # A x < b descript the unsafe set, violated if exist x such that max spec ai x - bi <= 0    
            for i in 1:batch_size
                CUDA.@allowscalar center_res[i] <= -tol && (results[i] = BasicResult(:violated))
                CUDA.@allowscalar spec_l[i] > -tol && (results[i] = BasicResult(:holds))
            end
        else # holds if forall x such that max spec ai x - bi <= tol
            for i in 1:batch_size
                # @show 1e4 * tol
                if CUDA.@allowscalar spec_u[i] <= tol #|| minimum(radius_hyperrectangle(batch_input[i].domain)) <= 1e-5
                    push!(results, BasicResult(:holds)) #results[i] = BasicResult(:holds)
                end
                # @show spec_l[i], spec_u[i]
                # CUDA.@allowscalar center_res[i] > tol && (results[i] = CounterExampleResult(:violated, center)) #BasicResult(:violated))
                if CUDA.@allowscalar center_res[i] > tol 
                    # @show center
                    push!(results, CounterExampleResult(:violated, center))
                    # push!(results, BasicResult(:unknown))
                    # results[i] = CounterExampleResult(:violated, center)
                else
                    push!(results, BasicResult(:unknown))
                end
            end
        end
    end
    
    return results
end

function Lie_IA_bound(prop_method::VeriGrad, batch_input::AbstractArray, bound::VeriGradBound)
    @assert size(bound.batch_Low, 3) == 1 "currently only support batchsize == 1"
    Lie_solver = Crown(prop_method.use_gpu, true, true, ModelVerification.zero_slope)
    # @show prop_method.Lie_linear_bound
    prop_method.Lie_linear_bound[1] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[1]) : prop_method.Lie_linear_bound[1]
    prop_method.Lie_linear_bound[2] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[2]) : prop_method.Lie_linear_bound[2]
    prop_method.Lie_linear_bound[3] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[3]) : prop_method.Lie_linear_bound[3]
    prop_method.Lie_linear_bound[4] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[4]) : prop_method.Lie_linear_bound[4]
    prop_method.Lie_linear_bound[5] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[5]) : prop_method.Lie_linear_bound[5]
    # @show bound.batch_Low[:, :, 1], bound.batch_Low[:, end, 1]
    bound_list = []
    bound1 = init_batch_bound(Lie_solver, batch_input, nothing)
    # @show bound1
    dense1 = Dense(size(bound.batch_Low, 2)-1, size(bound.batch_Low, 2)-1)
    dense1.weight .= -prop_method.Lie_linear_bound[1]
    dense1.bias .= -prop_method.Lie_linear_bound[3]
    bound1 = propagate_layer_batch(Lie_solver, dense1, bound1, nothing)
    # @show bound1
    bound1 = propagate_layer_batch(Lie_solver, relu, bound1, nothing)
    # @show bound1
    dense12 = Dense(size(bound.batch_Low, 2)-1, 1)
    dense12.weight .= -clamp.(bound.batch_Low[:, end, 1]', -Inf, 0)
    dense12.bias .= [0]
    
    bound1 = propagate_layer_batch(Lie_solver, dense12, bound1, nothing)
    push!(bound_list, bound1)
    # @show bound1

    # Lie_solver = Crown(prop_method.use_gpu, true, true, ModelVerification.zero_slope)
    bound2 = init_batch_bound(Lie_solver, batch_input, nothing)
    dense2 = Dense(size(bound.batch_Low, 2)-1, size(bound.batch_Low, 2)-1)
    dense2.weight .= prop_method.Lie_linear_bound[1]
    dense2.bias .= prop_method.Lie_linear_bound[3]
    bound2 = propagate_layer_batch(Lie_solver, dense2, bound2, nothing)
    bound2 = propagate_layer_batch(Lie_solver, relu, bound2, nothing)
    dense22 = Dense(size(bound.batch_Low, 2)-1, 1)
    dense22.weight .= clamp.(bound.batch_Up[:, end, 1]', -Inf, 0)
    dense22.bias .= [0]
    bound2 = propagate_layer_batch(Lie_solver, dense22, bound2, nothing)
    push!(bound_list, bound2)
    # @show bound2

    bound3 = init_batch_bound(Lie_solver, batch_input, nothing)
    dense3 = Dense(size(bound.batch_Low, 2)-1, size(bound.batch_Low, 2)-1)
    dense3.weight .= -prop_method.Lie_linear_bound[2]
    dense3.bias .= -prop_method.Lie_linear_bound[4]
    bound3 = propagate_layer_batch(Lie_solver, dense3, bound3, nothing)
    bound3 = propagate_layer_batch(Lie_solver, relu, bound3, nothing)
    dense32 = Dense(size(bound.batch_Low, 2)-1, 1)
    dense32.weight .= -clamp.(bound.batch_Low[:, end, 1]', 0, Inf)
    dense32.bias .= [0]
    bound3 = propagate_layer_batch(Lie_solver, dense32, bound3, nothing)
    push!(bound_list, bound3)

    bound4 = init_batch_bound(Lie_solver, batch_input, nothing)
    dense4 = Dense(size(bound.batch_Low, 2)-1, size(bound.batch_Low, 2)-1)
    dense4.weight .= prop_method.Lie_linear_bound[2]
    dense4.bias .= prop_method.Lie_linear_bound[4]
    bound4 = propagate_layer_batch(Lie_solver, dense4, bound4, nothing)
    bound4 = propagate_layer_batch(Lie_solver, relu, bound4, nothing)
    dense42 = Dense(size(bound.batch_Low, 2)-1, 1)
    dense42.weight .= clamp.(bound.batch_Up[:, end, 1]', 0, Inf)
    dense42.bias .= [0]
    bound4 = propagate_layer_batch(Lie_solver, dense42, bound4, nothing)
    push!(bound_list, bound4)
    # @show bound1
    # @show bound2
    # @show bound3
    # @show bound4
    # @show bound_list
    # bound = CrownBound(sum([bound_list[1].batch_Low for i in 1:4]),sum([bound_list[1].batch_Up for i in 1:4]),  bound1.batch_data_min, bound1.batch_data_max, bound1.img_size)
    # @show bound1.batch_data_min, bound1.batch_data_max
    bound = CrownBound(bound_list[1].batch_Low+bound_list[2].batch_Low+bound_list[3].batch_Low+bound_list[4].batch_Low,
                            bound_list[1].batch_Up+bound_list[2].batch_Up+bound_list[3].batch_Up+bound_list[4].batch_Up,  bound1.batch_data_min, bound1.batch_data_max, bound1.img_size)
    return bound
end

# function propagate_layer_batch_Lie(prop_method::BetaCrown, layer::Dense, bound::BetaCrownBound, batch_info)
#     #TO DO: we haven't consider the perturbation in weight and bias
#     bias_lb =layer.bias
#     lA_W = uA_W = nothing 
#     # println("=== in dense ===")
#     # println("bound.lower_A_x: ", bound.lower_A_x)
#     # @assert !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
#     weight = prop_method.use_gpu ? fmap(cu, layer.weight) : layer.weight
#     bias = bias_lb
#     lA_x = prop_method.bound_lower ? dense_bound_oneside(weight, bias, 1) : nothing
#     uA_x = prop_method.bound_upper ? dense_bound_oneside(weight, bias, 1) : nothing

#     New_bound = BetaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
#     return New_bound
#     # end
# end

# function process_bound_Lie(prop_method::BetaCrown, batch_bound::BetaCrownBound, batch_out_spec, model_info, batch_info)
    
#     to = get_timer("Shared")
#     # @show size(batch_bound.batch_data_min)
#     @timeit to "compute_bound" compute_bound_beta = Compute_bound(batch_bound.batch_data_min, batch_bound.batch_data_max)
#     #bound_model = Chain(push!(prop_method.bound_lower ? batch_bound.lower_A_x : batch_bound.upper_A_x, compute_bound_beta)) 
#     # println("batch_bound.lower_A_x: ", length(batch_bound.lower_A_x))
#     # println("batch_bound.upper_A_x: ", length(batch_bound.upper_A_x))

#     # @show batch_bound.img_size

#     # for node in model_info.all_nodes
#     #     @show node
#     #     @show typeof(batch_info[node][:bound])
#     # end

#     # for i in eachindex(batch_bound.lower_A_x)
#     #     @show i, typeof(batch_bound.lower_A_x), typeof(batch_bound.lower_A_x[i])
#     # end
#     # @show length(batch_bound.lower_A_x)
#     # for op in batch_bound.lower_A_x
#     #     @show typeof(op)
#     #     @show Flux.params(op)
#     # end
#     # @show Flux.params(batch_bound.lower_A_x)
#     # @show Flux.params(batch_bound.upper_A_x)
#     get_lower_A = bound -> bound.lower_A_x
#     get_upper_A = bound -> bound.upper_A_x
    
#     bound_lower_model = build_bound_graph(prop_method, model_info, batch_info, get_lower_A)
#     bound_upper_model = build_bound_graph(prop_method, model_info, batch_info, get_upper_A)
#     # @show model_info.all_nodes
#     # for op in bound_lower_model
#     #     @show typeof(op)
#     # end
#     # println("====")
#     # x = batch_info[:spec_A_b]
#     # @show length(bound_lower_model)
#     # @show size(x[1]), size(x[2])
#     # for op in bound_lower_model
#     #     @show typeof(op)
#     #     x = op(x)
#     #     @show size(x[1]), size(x[2])
#     # end
#     # @show bound_lower_model

#     bound_lower_model = Chain(push!(bound_lower_model, compute_bound_beta)) 
#     bound_upper_model = Chain(push!(bound_upper_model, compute_bound_beta)) 

#     bound_lower_model = prop_method.use_gpu ? bound_lower_model |> gpu : bound_lower_model
#     bound_upper_model = prop_method.use_gpu ? bound_upper_model |> gpu : bound_upper_model

#     # @show Flux.params(bound_upper_model)
#     # for polytope output set, spec holds if upper bound of (spec_A x - b) < 0 for all dimension. therefore minimize maximum(spec_A x - b)
#     # for complement polytope set, spec holds if lower bound of (spec_A x - b) > 0 for any dimension. therefore maximize maximum(spec_A x - b), that is minimize -maximum(spec_A x - b)
    
#     # After conversion, we only need to decide if lower bound of spec_A y-spec_b > 0 or if upper bound of spec_A y - spec_b < 0
#     # The new out is spec_A*y-b, whose dimension is spec_dim x batch_size.
#     # Therefore, we set new_spec_A: 1(new_spec_dim) x original_spec_dim x batch_size, new_spec_b: 1(new_spec_dim) x batch_size,
#     # spec_dim, out_dim, batch_size = size(out_specs.A)
#     # out_specs = LinearSpec(ones((1, spec_dim, batch_size)), zeros(1, batch_size), out_specs.is_complement)

#     # @timeit to "optimize_model" bound_lower_model = optimize_model(bound_lower_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
#     # @timeit to "optimize_model" bound_upper_model = optimize_model(bound_upper_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
    
#     if length(Flux.params(bound_lower_model)) > 0
#         loss_func = l_u -> -sum(l_u[1]) # surrogate loss to maximize the min spec
#         # loss_func = l_u -> -sum(exp.(l_u[1])) # surrogate loss to maximize the min spec
#         # loss_func = l_u -> -sum(l_u[1].^3) # surrogate loss to maximize the min spec
#         @timeit to "optimize_model" bound_lower_model = optimize_model(bound_lower_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
#     end
#     if length(Flux.params(bound_upper_model)) > 0
#         # @show "optimize"
#         loss_func = l_u -> sum(l_u[2]) # surrogate loss to minimize the max spec
#         # loss_func = l_u -> sum(exp.(l_u[2])) # surrogate loss to minimize the max spec
#         # loss_func = l_u -> sum(l_u[2].^3) # surrogate loss to minimize the max spec
#         # loss_func = l_u -> -sum(l_u[2]) # surrogate loss to minimize the max spec
#         @timeit to "optimize_model" bound_upper_model = optimize_model(bound_upper_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
#     end

#     # print_beta_layers(bound_lower_model, batch_info[:init_A_b])

#     # println("manual set")
#     # print_beta_layers(bound_lower_model, batch_info[:init_A_b])
#     # lower_l, lower_u = bound_lower_model(batch_info[:init_A_b] |> gpu)
#     # @timeit to "optimize_model" bound_lower_model = optimize_model(bound_lower_model, batch_info[:init_A_b] |> gpu, loss_func, prop_method.optimizer, 1)
#     # println("lower bound")
#     # println(lower_l, " ", lower_u)
    
#     if prop_method.use_alpha # remember the optimized alpha for warm starting sub-branches.
#         [batch_info[l.node][:lower_bound_alpha] = l.alpha for l in Flux.modules(bound_lower_model) if l isa BetaLayer]
#         [batch_info[l.node][:upper_bound_alpha] = l.alpha for l in Flux.modules(bound_upper_model) if l isa BetaLayer]
#     end
#     if prop_method.use_beta # remember the optimized beta for warm starting sub-branches.
#         [batch_info[l.node][:beta_lower] = l.beta for l in Flux.modules(bound_lower_model) if l isa BetaLayer]
#         [batch_info[l.node][:beta_upper] = l.beta for l in Flux.modules(bound_upper_model) if l isa BetaLayer]
#     end

#     # @show size(batch_info[:spec_A_b][1]), size(batch_info[:spec_A_b][2])

#     lower_spec_l, lower_spec_u = bound_lower_model(batch_info[:spec_A_b])
#     upper_spec_l, upper_spec_u = bound_upper_model(batch_info[:spec_A_b])
#     # @show lower_spec_l
#     # @show lower_spec_u
#     # @show upper_spec_l
#     # @show upper_spec_u
#     # println("spec")

#     # The symbolic beta-crown bound can be transferred into crown bound form
#     crown_bound_sym = to_crown_bound(Chain(bound_lower_model.layers[1:end-1]...),Chain(bound_upper_model.layers[1:end-1]...), batch_info[:spec_A_b], batch_bound.batch_data_min, batch_bound.batch_data_max, batch_bound.img_size)
#     # @show compute_bound(crown_bound_sym)
#     node = start_nodes(prop_method, model_info)[1]
#     # @show model_info.all_nodes
#     # @show prev_nodes(prop_method, model_info, node)
#     # @show next_node = prev_nodes(prop_method, model_info, node)
#     # @show model_info.activation_nodes
#     # @assert (next_node in model_info.activation_nodes)
#     batch_info[node][:symbolic_crown_bound] = crown_bound_sym 

    
#     # spec_bound_lower = batch_out_spec.is_complement ? true : false
#     # spec_bound_upper = batch_out_spec.is_complement ? false : true
#     # if spec_bound_lower
#     #     #batch_info = get_pre_relu_A(init, prop_method.use_gpu, true, model_info, batch_info)
#     #     @timeit to "get_pre_relu_spec_A" batch_info = get_pre_relu_spec_A(batch_info[:spec_A_b], prop_method.use_gpu, true, model_info, batch_info)
#     # end
#     # if spec_bound_upper
#     #     #batch_info = get_pre_relu_A(init, prop_method.use_gpu, false, model_info, batch_info)
#     #     @timeit to "get_pre_relu_spec_A" batch_info = get_pre_relu_spec_A(batch_info[:spec_A_b], prop_method.use_gpu, false, model_info, batch_info)
#     # end
#     return ConcretizeCrownBound(lower_spec_l, upper_spec_u, batch_bound.batch_data_min, batch_bound.batch_data_max), batch_info
# end
# function propagate_layer_batch_Lie(prop_method::BetaCrown, layer::typeof(relu), bound::BetaCrownBound, batch_info)
#     # @show "========= relu"
#     node = batch_info[:current_node]
#     #= if !haskey(batch_info[node], :pre_lower) || !haskey(batch_info[node], :pre_upper)
#         lower, upper = compute_bound(batch_info[node][:pre_bound])
#         batch_info[node][:pre_lower] = lower
#         batch_info[node][:pre_upper] = upper
#     else =#
#     lower = batch_info[node][:pre_lower]
#     upper = batch_info[node][:pre_upper]
#     #end
#     # println("=== in relu ===")
#     # println("lower: ", lower)
#     # println("upper: ", upper)
#     # @show batch_info[node][:lower_bound_alpha]
#     # @show batch_info[node][:upper_bound_alpha]

#     lower_bound_alpha = batch_info[node][:lower_bound_alpha]
#     upper_bound_alpha = batch_info[node][:upper_bound_alpha]
#     upper_slope, upper_bias = relu_upper_bound(lower, upper) #upper_slope:upper of slope  upper_bias:Upper of bias
#     lower_bias = prop_method.use_gpu ? fmap(cu, zeros(size(upper_bias))) : zeros(size(upper_bias))

#     active_mask = (lower .>= 0)
#     inactive_mask = (upper .<= 0)
#     unstable_mask = (upper .> 0) .& (lower .< 0)
#     batch_info[node][:unstable_mask] = unstable_mask

#     beta_lower = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_lower]) : batch_info[node][:beta_lower]
#     beta_upper = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_upper]) : batch_info[node][:beta_upper]
#     beta_lower_index = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_lower_index]) : batch_info[node][:beta_lower_index]
#     beta_upper_index = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_upper_index]) : batch_info[node][:beta_upper_index]
#     beta_lower_S = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_lower_S]) : batch_info[node][:beta_lower_S]
#     beta_upper_S = prop_method.use_gpu ? fmap(cu, batch_info[node][:beta_upper_S]) : batch_info[node][:beta_upper_S]

#     # lower_A = deepcopy(bound.lower_A_x)
#     # upper_A = deepcopy(bound.upper_A_x)

#     # println("lower_A: ", lower_A)
#     # println("before lower_A: ")
#     # print_beta_layers(lower_A, batch_info[:init_A_b])
#     # println("before upper_A: ")
#     # print_beta_layers(upper_A, batch_info[:init_A_b])
#     # @show "setting ", node
#     batch_info[node][:pre_upper_A_function] = nothing
#     batch_info[node][:pre_lower_A_function] = nothing

#     Beta_Lower_Layer = prop_method.bound_lower ? BetaLayer(node, lower_bound_alpha, beta_lower, beta_lower_S, beta_lower_index, batch_info[:spec_A_b], true, unstable_mask, active_mask, upper_slope, lower_bias, upper_bias, prop_method.use_alpha, prop_method.use_beta) : nothing
#     # if 
#         # batch_info[node][:pre_lower_A_function] = copy(lower_A)
#         # println("Beta_Lower_Layer.beta_lower: ", Beta_Lower_Layer.beta)
#         # push!(lower_A, Beta_Lower_Layer)
#         # @show Flux.params(Beta_Lower_Layer)
#     # end

#     Beta_Upper_Layer = prop_method.bound_upper ? BetaLayer(node, upper_bound_alpha, beta_upper, beta_upper_S, beta_upper_index, batch_info[:spec_A_b], false, unstable_mask, active_mask, upper_slope, lower_bias, upper_bias, false, prop_method.use_beta) : nothing
#     # if prop_method.bound_upper
#         # batch_info[node][:pre_upper_A_function] = copy(upper_A)
#         # println("Beta_Upper_Layer.beta_lower: ", Beta_Upper_Layer.beta)
#         # push!(upper_A, Beta_Upper_Layer)
#     # end
#     push!(batch_info[:Beta_Lower_Layer_node], node)
#     New_bound = BetaCrownBound(Beta_Lower_Layer, Beta_Upper_Layer, nothing, nothing, bound.batch_data_min, bound.batch_data_max, bound.img_size)

#     # @show Flux.params(lower_A)
#     # println("lower_A: ", lower_A)
#     # println("after lower_A: ")
#     # print_beta_layers(lower_A, batch_info[:init_A_b])
#     # println("after upper_A: ")
#     # print_beta_layers(upper_A, batch_info[:init_A_b])

#     return New_bound
# end

# function Lie_IA_bound_beta(prop_method::VeriGrad, batch_input::AbstractArray, bound::VeriGradBound)
#     @assert size(bound.batch_Low, 3) == 1 "currently only support batchsize == 1"
#     Lie_solver = BetaCrown(false, false, false, nothing, true, true, Flux.ADAM(0.1), 10, true)
#     # @show prop_method.Lie_linear_bound
#     prop_method.Lie_linear_bound[1] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[1]) : prop_method.Lie_linear_bound[1]
#     prop_method.Lie_linear_bound[2] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[2]) : prop_method.Lie_linear_bound[2]
#     prop_method.Lie_linear_bound[3] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[3]) : prop_method.Lie_linear_bound[3]
#     prop_method.Lie_linear_bound[4] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[4]) : prop_method.Lie_linear_bound[4]
#     prop_method.Lie_linear_bound[5] = prop_method.use_gpu ? fmap(cu, prop_method.Lie_linear_bound[5]) : prop_method.Lie_linear_bound[5]
#     # @show bound.batch_Low[:, :, 1], bound.batch_Low[:, end, 1]
#     bound_list = []
#     bound1 = init_batch_bound(Lie_solver, batch_input, nothing)

#     dense12 = Dense(size(bound.batch_Low, 2)-1, 1)
#     dense12.weight .= -clamp.(bound.batch_Low[:, end, 1]', -Inf, 0)
#     dense12.bias .= [0]
#     bound1 = propagate_layer_batch_Lie(Lie_solver, dense12, bound1, nothing)

#     bound1 = propagate_layer_batch(Lie_solver, relu, bound1, nothing)
    
#     dense1 = Dense(size(bound.batch_Low, 2)-1, size(bound.batch_Low, 2)-1)
#     dense1.weight .= -prop_method.Lie_linear_bound[1]
#     dense1.bias .= -prop_method.Lie_linear_bound[3]
#     bound1 = propagate_layer_batch(Lie_solver, dense1, bound1, nothing)
#     # @show bound1
#     bound1 = propagate_layer_batch(Lie_solver, relu, bound1, nothing)
#     # @show bound1
    
    
    
#     push!(bound_list, bound1)
#     # @show bound1

#     bound1 = init_batch_bound(Lie_solver, batch_input, nothing)
#     dense1 = Dense(size(bound.batch_Low, 2)-1, size(bound.batch_Low, 2)-1)
#     dense1.weight .= prop_method.Lie_linear_bound[1]
#     dense1.bias .= prop_method.Lie_linear_bound[3]
#     bound1 = propagate_layer_batch(Lie_solver, dense1, bound1, nothing)
#     bound1 = propagate_layer_batch(Lie_solver, relu, bound1, nothing)
#     dense12 = Dense(size(bound.batch_Low, 2)-1, 1)
#     dense12.weight .= clamp.(bound.batch_Up[:, end, 1]', -Inf, 0)
#     dense12.bias .= [0]
#     bound1 = propagate_layer_batch(Lie_solver, dense12, bound1, nothing)
#     push!(bound_list, bound1)

#     bound1 = init_batch_bound(Lie_solver, batch_input, nothing)
#     dense1 = Dense(size(bound.batch_Low, 2)-1, size(bound.batch_Low, 2)-1)
#     dense1.weight .= -prop_method.Lie_linear_bound[2]
#     dense1.bias .= -prop_method.Lie_linear_bound[4]
#     bound1 = propagate_layer_batch(Lie_solver, dense1, bound1, nothing)
#     bound1 = propagate_layer_batch(Lie_solver, relu, bound1, nothing)
#     dense12 = Dense(size(bound.batch_Low, 2)-1, 1)
#     dense12.weight .= -clamp.(bound.batch_Low[:, end, 1]', 0, Inf)
#     dense12.bias .= [0]
#     bound1 = propagate_layer_batch(Lie_solver, dense12, bound1, nothing)
#     push!(bound_list, bound1)

#     bound1 = init_batch_bound(Lie_solver, batch_input, nothing)
#     dense1 = Dense(size(bound.batch_Low, 2)-1, size(bound.batch_Low, 2)-1)
#     dense1.weight .= prop_method.Lie_linear_bound[2]
#     dense1.bias .= prop_method.Lie_linear_bound[4]
#     bound1 = propagate_layer_batch(Lie_solver, dense1, bound1, nothing)
#     bound1 = propagate_layer_batch(Lie_solver, relu, bound1, nothing)
#     dense12 = Dense(size(bound.batch_Low, 2)-1, 1)
#     dense12.weight .= clamp.(bound.batch_Up[:, end, 1]', 0, Inf)
#     dense12.bias .= [0]
#     bound1 = propagate_layer_batch(Lie_solver, dense12, bound1, nothing)
#     push!(bound_list, bound1)
#     # @show bound_list
#     # bound = CrownBound(sum([bound_list[1].batch_Low for i in 1:4]),sum([bound_list[1].batch_Up for i in 1:4]),  bound1.batch_data_min, bound1.batch_data_max, bound1.img_size)
#     # @show bound1.batch_data_min, bound1.batch_data_max
#     bound = CrownBound(bound_list[1].batch_Low+bound_list[2].batch_Low+bound_list[3].batch_Low+bound_list[4].batch_Low,
#                             bound_list[1].batch_Up+bound_list[2].batch_Up+bound_list[3].batch_Up+bound_list[4].batch_Up,  bound1.batch_data_min, bound1.batch_data_max, bound1.img_size)
#     return bound
# end

function find_crown_bound(nn_model::Chain, batch_input::AbstractArray)
    solver = Crown(false, true, true, ModelVerification.zero_slope)
    bound = ModelVerification.init_batch_bound(solver, batch_input, nothing)
    # @show bound1
    for layer in nn_model
        bound = ModelVerification.propagate_layer_batch(solver, layer, bound, nothing)
        if isa(layer.σ, typeof(relu))
            bound = ModelVerification.propagate_layer_batch(solver, relu, bound, nothing)
        end
    end
    return bound
end
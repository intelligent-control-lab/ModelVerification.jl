"""
    BetaCrown <: BatchBackwardProp
"""
mutable struct BetaCrown <: BatchBackwardProp 
    use_alpha::Bool
    use_beta::Bool
    use_gpu::Bool
    pre_bound_method::Union{BatchForwardProp, BatchBackwardProp, Nothing, Dict}
    bound_lower::Bool
    bound_upper::Bool
    optimizer
    train_iteration::Int
    inherit_pre_bound::Bool
end
BetaCrown(nothing) = BetaCrown(true, true, true, nothing, true, true, Flux.ADAM(0.1), 10, true)
BetaCrown(;use_alpha=true, use_beta=true, use_gpu=true, pre_bound_method=BetaCrown(nothing), bound_lower=true, bound_upper=true, optimizer=Flux.ADAM(0.1), train_iteration=10, inherit_pre_bound=true) =
    BetaCrown(use_alpha, use_beta, use_gpu, pre_bound_method, bound_lower, bound_upper, optimizer, train_iteration, inherit_pre_bound)

"""
    BetaCrownBound <: Bound
"""
struct BetaCrownBound <: Bound
    lower_A_x
    upper_A_x
    lower_A_W
    upper_A_W
    batch_data_min
    batch_data_max
end

"""
    Compute_bound
"""
struct Compute_bound
    batch_data_min
    batch_data_max
end
Flux.@functor Compute_bound ()


function (f::Compute_bound)(x)
    #z = zeros(size(x[1]))
    #l = batched_vec(max.(x[1], z), f.batch_data_min) + batched_vec(min.(x[1], z), f.batch_data_max) .+ x[2]
    #u = batched_vec(max.(x[1], z), f.batch_data_max) + batched_vec(min.(x[1], z), f.batch_data_min) .+ x[2]
    A_pos = clamp.(x[1], 0, Inf)
    A_neg = clamp.(x[1], -Inf, 0)
    l = batched_vec(A_pos, f.batch_data_min) + batched_vec(A_neg, f.batch_data_max) .+ x[2]
    u = batched_vec(A_pos, f.batch_data_max) + batched_vec(A_neg, f.batch_data_min) .+ x[2]
    return l, u
end 


"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::BetaCrown, problem::Problem)
"""
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::BetaCrown, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    model = prop_method.use_gpu ? problem.Flux_model |> gpu : problem.Flux_model
    return model_info, Problem(problem.onnx_model_path, model, init_bound(prop_method, problem.input), problem.output)
end

"""
    init_batch_bound(prop_method::BetaCrown, batch_input::AbstractArray, batch_output::LinearSpec)
"""
function init_batch_bound(prop_method::BetaCrown, batch_input::AbstractArray, batch_output::LinearSpec)
    batch_data_min = prop_method.use_gpu ? fmap(cu, cat([low(h.domain) for h in batch_input]..., dims=2)) : cat([low(h.domain) for h in batch_input]..., dims=2)
    batch_data_max = prop_method.use_gpu ? fmap(cu, cat([high(h.domain) for h in batch_input]..., dims=2)) : cat([high(h.domain) for h in batch_input]..., dims=2)
    bound = BetaCrownBound([], [], nothing, nothing, batch_data_min, batch_data_max)
    return bound
end

"""
    prepare_method(prop_method::BetaCrown, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info)
"""
function prepare_method(prop_method::BetaCrown, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info)
    out_specs = get_linear_spec(batch_output)

    if prop_method.use_gpu
        out_specs = LinearSpec(fmap(cu, out_specs.A), fmap(cu, out_specs.b), fmap(cu, out_specs.is_complement))
    end
    return prepare_method(prop_method, batch_input, out_specs, batch_inheritance, model_info)
end

function batchify_inheritance(prop_method::BetaCrown, inheritance_list::AbstractVector, model_info)
    eltype(inheritance_list) == Nothing && return nothing
    batch_inheritance = Dict()
    for node in model_info.activation_nodes
        l = cat([ih[node][:pre_lower] for ih in inheritance_list]..., dims=2)
        u = cat([ih[node][:pre_upper] for ih in inheritance_list]..., dims=2)
        # println("size(ih[node][:pre_upper]): ", size(inheritance_list[1][node][:pre_upper]))
        # println("size(l): ", size(l))
        batch_inheritance[node] = Dict(
            :pre_lower => prop_method.use_gpu ? l |> gpu : l,
            :pre_upper => prop_method.use_gpu ? u |> gpu : u
        )
    end
    # length(batch_inheritance) == 0 && return nothing
    return batch_inheritance
end

"""
    prepare_method(prop_method::BetaCrown, batch_input::AbstractVector, out_specs::LinearSpec, model_info)
"""
function prepare_method(prop_method::BetaCrown, batch_input::AbstractVector, out_specs::LinearSpec, inheritance_list::AbstractVector, model_info)
    
    batch_size = length(batch_input)
    
    batch_info = init_propagation(prop_method, batch_input, out_specs, model_info)
    batch_info[:spec_A_b] = [out_specs.A, .-out_specs.b] # spec_A x < spec_b  ->  A x + b < 0, need negation

    batch_inheritance = batchify_inheritance(prop_method, inheritance_list, model_info)

    println("batch_inheritance: ", batch_inheritance)

    if prop_method.inherit_pre_bound && !isnothing(batch_inheritance) # pre_bound can be inherited from the parent branch 
        println("inheritating pre bound ...")
        for node in model_info.activation_nodes
            println("batch_inheritance[node][:pre_lower]:", batch_inheritance[node][:pre_lower])
            batch_info[node][:pre_lower] = batch_inheritance[node][:pre_lower]
            batch_info[node][:pre_upper] = batch_inheritance[node][:pre_upper]
        end
    elseif prop_method.pre_bound_method isa BetaCrown  # requires recursive bounding, iterate from first layer
        println("computing pre bound ...")
        # need forward BFS to compute pre_bound of all, 
        pre_bounds = Dict()
        for node in model_info.activation_nodes
            println("node: ", node)
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            
            sub_model_info = get_sub_model(model_info, prev_node)
            n_out = size(model_info.node_layer[prev_node].weight)[1]
            
            I_spec = LinearSpec(repeat(Matrix(1.0I, n_out, n_out),1,1,batch_size), zeros(n_out, batch_size), false)
            if prop_method.use_gpu
                I_spec = LinearSpec(fmap(cu, I_spec.A), fmap(cu, I_spec.b), fmap(cu, I_spec.is_complement))
            end
            
            sub_out_spec, sub_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, I_spec, [pre_bounds], sub_model_info)
            # println("keys: ", keys(sub_batch_info))
            if haskey(sub_batch_info, "dense_0_relu")
                println("dense_0_relu low A:", sub_batch_info["dense_0_relu"])
            end
            # println("dense_0_relu low A:", sub_batch_info["dense_0_relu"].lower_A_x)
            sub_batch_bound, sub_batch_info = propagate(prop_method.pre_bound_method, sub_model_info, sub_batch_info)
            # println("1 sub_batch_bound:", typeof(sub_batch_bound.lower_A_x))
            # println("sub_batch_bound.lower_A_x: ", sub_batch_bound.lower_A_x)
            # println("sub_batch_bound.upper_A_x: ", sub_batch_bound.upper_A_x)
            sub_batch_bound, sub_batch_info = process_bound(prop_method.pre_bound_method, sub_batch_bound, sub_out_spec, sub_model_info, sub_batch_info)
            # println("2 sub_batch_bound:", typeof(sub_batch_bound))
            l, u = compute_bound(sub_batch_bound) # reach_dim x batch 

            batch_info[node][:pre_lower], batch_info[node][:pre_upper] = l, u
            pre_bounds[node] = Dict(:pre_lower => l, :pre_upper => u)

        end
    elseif !isnothing(prop_method.pre_bound_method)
        pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, out_specs, [nothing], model_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            pre_bound = pre_batch_info[prev_node][:bound]
            batch_info[node][:pre_lower], batch_info[node][:pre_upper] = compute_bound(pre_bound) # reach_dim x batch 
            # println("assigning", node," ", prev_node)
        end
    end
    
    batch_info[:batch_size] = length(batch_input)
    for node in model_info.all_nodes
        batch_info[node][:beta] = 1
        batch_info[node][:max_split_number] = 1
        batch_info[node][:weight_ptb] = false
        batch_info[node][:bias_ptb] = false
    end
    #initialize alpha & beta
    # println("model_info.activation_nodes")
    # println(model_info.activation_nodes)
    # @assert false
    for node in model_info.activation_nodes
        batch_info = init_alpha(model_info.node_layer[node], node, batch_info, batch_input)
        batch_info = init_beta(model_info.node_layer[node], node, batch_info, batch_input)
    end
    n = size(out_specs.A, 2)
    batch_info[:init_A_b] = init_A_b(n, batch_info[:batch_size])
    batch_info[:Beta_Lower_Layer_node] = []#store the order of the node which has AlphaBetaLayer
    return out_specs, batch_info
end 

"""
    get_inheritance(prop_method::BetaCrown, batch_info::Dict, batch_idx::Int)

Extract useful informations from batch_info.
These information will later be inheritated by the new branch created by split.

## Arguments
- `prop_method` (`ForwardProp`): Solver being used.
- `batch_info` (`Dict`): all the information collected in propagation.
- `batch_idx`: the index of the interested branch in the batch.

## Returns
- `inheritance`: a dict that contains all the information will be inheritated.
"""
function get_inheritance(prop_method::BetaCrown, batch_info::Dict, batch_idx::Int, model_info)
    prop_method.inherit_pre_bound || return nothing
    inheritance = Dict()
    # println("batch_info")
    # println(keys(batch_info))
    for node in model_info.activation_nodes
        # println(size(batch_info[node][:pre_lower]))
        inheritance[node] = Dict(
            :pre_lower => batch_info[node][:pre_lower][:,batch_idx],
            :pre_upper => batch_info[node][:pre_upper][:,batch_idx]
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
    init_alpha(layer::typeof(relu), node, batch_info, batch_input)
"""
function init_alpha(layer::typeof(relu), node, batch_info, batch_input)
    
    # relu_input_lower, relu_input_upper = update_bound_by_relu_con(node, batch_input, relu_input_lower, relu_input_upper)

    # println("relu_input_lower: ", relu_input_lower)
    # println("relu_input_upper: ", relu_input_upper)

    # relu_input_lower, relu_input_upper = compute_bound(batch_info[node][:pre_bound]) # reach_dim x batch 
    # batch_info[node][:pre_lower] = relu_input_lower # reach_dim x batch 
    # batch_info[node][:pre_upper] = relu_input_upper

    #batch_size = size(relu_input_lower)[end]
    l = batch_info[node][:pre_lower]
    u = batch_info[node][:pre_upper]

    unstable_mask = (u .> 0) .& (l .< 0) #indices of non-zero alphas/ indices of activative neurons
    alpha_indices = findall(unstable_mask) 
    upper_slope, upper_bias = relu_upper_bound(l, u) #upper slope and upper bias
    # lower_slope = convert(typeof(upper_slope), upper_slope .> 0.5) #lower slope
    lower_slope = copy(upper_slope) #lower slope
    #lower_slope = zeros(size(upper_slope))
    #minimum_sparsity = batch_info[node]["minimum_sparsity"]
    #total_neuron_size = length(l) ÷ batch_size #number of the neuron of the pre_layer of relu

    #fully alpha
    @assert ndims(l) == 2 || ndims(l) == 4 "pre_layer of relu should be dense or conv"
    #if(ndims(l) == 2) #pre_layer of relu is dense 
    #end
    #alpha_lower is for lower bound, alpha_upper is for upper bound
    alpha_lower = lower_slope .* unstable_mask
    alpha_upper = upper_slope .* unstable_mask
    batch_info[node][:alpha_lower] = alpha_lower #reach_dim x batch
    batch_info[node][:alpha_upper] = alpha_upper #reach_dim x batch

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
    init_bound(prop_method::BetaCrown, input) 
"""
function init_bound(prop_method::BetaCrown, input) 
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
                # println(string(nameof(typeof(layer))))
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
    process_bound(prop_method::BetaCrown, batch_bound::BetaCrownBound, batch_out_spec, model_info, batch_info)
"""
function process_bound(prop_method::BetaCrown, batch_bound::BetaCrownBound, batch_out_spec, model_info, batch_info)
    to = get_timer("Shared")
    @timeit to "compute_bound" compute_bound = Compute_bound(batch_bound.batch_data_min, batch_bound.batch_data_max)
    #bound_model = Chain(push!(prop_method.bound_lower ? batch_bound.lower_A_x : batch_bound.upper_A_x, compute_bound)) 
    println("batch_bound.lower_A_x: ", length(batch_bound.lower_A_x))
    println("batch_bound.upper_A_x: ", length(batch_bound.upper_A_x))
    bound_lower_model = Chain(push!(batch_bound.lower_A_x, compute_bound)) 
    bound_upper_model = Chain(push!(batch_bound.upper_A_x, compute_bound)) 
    bound_lower_model = prop_method.use_gpu ? bound_lower_model |> gpu : bound_lower_model
    bound_upper_model = prop_method.use_gpu ? bound_upper_model |> gpu : bound_upper_model
    # loss_func = prop_method.bound_lower ?  x -> - sum(x[1]) : x -> sum(x[2])

    # for polytope output set, spec holds if upper bound of (spec_A x - b) < 0 for all dimension. therefore minimize maximum(spec_A x - b)
    # for complement polytope set, spec holds if lower bound of (spec_A x - b) > 0 for any dimension. therefore maximize maximum(spec_A x - b), that is minimize -maximum(spec_A x - b)
    
    # After conversion, we only need to decide if lower bound of spec_A y-spec_b > 0 or if upper bound of spec_A y - spec_b < 0
    # The new out is spec_A*y-b, whose dimension is spec_dim x batch_size.
    # Therefore, we set new_spec_A: 1(new_spec_dim) x original_spec_dim x batch_size, new_spec_b: 1(new_spec_dim) x batch_size,
    # spec_dim, out_dim, batch_size = size(out_specs.A)
    # out_specs = LinearSpec(ones((1, spec_dim, batch_size)), zeros(1, batch_size), out_specs.is_complement)

    # loss_func = prop_method.bound_lower ?  x -> -maximum(x[1]) : x -> maximum(x[2]) # maximum leads to error in flux
    # loss_func = prop_method.bound_lower ?  x -> - sum(x[1]) : x -> sum(x[2])
    loss_func = prop_method.bound_lower ?  x -> -sum(x[1].^2) : x -> sum(x[2].^2) # surrogate loss to minimize the max spec

    # @timeit to "optimize_model" bound_lower_model = optimize_model(bound_lower_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
    # @timeit to "optimize_model" bound_upper_model = optimize_model(bound_upper_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)

    if length(Flux.params(bound_lower_model)) > 0
        @timeit to "optimize_model" bound_lower_model = optimize_model(bound_lower_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
    end
    if length(Flux.params(bound_upper_model)) > 0
        @timeit to "optimize_model" bound_upper_model = optimize_model(bound_upper_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
    end


    # result = bound_lower_model(batch_info[:init_A_b] |> gpu) 
    # loss = loss_func(result)
    # println("result: ", result)
    # println("loss: ", loss)
    
    # print_beta_layers(bound_lower_model, batch_info[:init_A_b])

    # println("manual set")
    # print_beta_layers(bound_lower_model, batch_info[:init_A_b])
    # lower_l, lower_u = bound_lower_model(batch_info[:init_A_b] |> gpu)
    # @timeit to "optimize_model" bound_lower_model = optimize_model(bound_lower_model, batch_info[:init_A_b] |> gpu, loss_func, prop_method.optimizer, 1)
    # println("lower bound")
    # println(lower_l, " ", lower_u)
    
    # println("bound_upper_model")
    
    # upper_l, upper_u = bound_upper_model(batch_info[:init_A_b] |> gpu)
    # println("upper bound")
    # println(upper_l, " ", upper_u)
    # println("=======================")

    for (index, params) in enumerate(Flux.params(bound_lower_model))
        relu_node = batch_info[:Beta_Lower_Layer_node][ceil(Int, index / 2)]
        if index % 2 == 1
            batch_info[relu_node][:alpha_lower] = params
        else
            batch_info[relu_node][:beta_lower] = params
        end
    end
    for (index, params) in enumerate(Flux.params(bound_upper_model))
        relu_node = batch_info[:Beta_Lower_Layer_node][ceil(Int, index / 2)]
        if index % 2 == 1
            batch_info[relu_node][:alpha_upper] = params
        else
            batch_info[relu_node][:beta_upper] = params
        end
    end

    
    lower_spec_l, lower_spec_u = bound_lower_model(batch_info[:spec_A_b])
    upper_spec_l, upper_spec_u = bound_upper_model(batch_info[:spec_A_b])
    # println("spec")
    # println("batch_bound.lower_A_x")
    # println(batch_bound.lower_A_x)
    # println("batch_bound.upper_A_x")
    # println(batch_bound.upper_A_x)
    # if isa(bound_lower_model[2], BetaLayer)
        # println("-----------------------")
        # print_beta_layers(bound_lower_model, batch_info[:spec_A_b])
        # println("-----------------------")
        # print_beta_layers(bound_upper_model, batch_info[:spec_A_b])
        # println("-----------------------")
        # bound_upper_model[2].alpha .= [0.444,0.444] |> gpu
        # println(bound_lower_model[2])
        # result = bound_lower_model(batch_info[:init_A_b] |> gpu) 
        # loss = loss_func(result)
        # println("result: ", result)
        # println("loss: ", loss)
        # println(bound_lower_model[2])
        # @assert false
        # println("lower alpha")
        # println(bound_lower_model[2].alpha)
        # println("lower_spec_l")
        # println(lower_spec_l)
        # println("upper alpha")
        # println(bound_upper_model[2].alpha)
        # println("upper_spec_u")
        # println(upper_spec_u)
    # end

    
    # for polytope output set, spec holds if upper bound of (spec_A x - b) < 0 for all dimension.
    # for complement polytope set, spec holds if lower bound of (spec_A x - b) > 0 for any dimension.
    
    spec_bound_lower = batch_out_spec.is_complement ? true : false
    spec_bound_upper = batch_out_spec.is_complement ? false : true
    if spec_bound_lower
        #batch_info = get_pre_relu_A(init, prop_method.use_gpu, true, model_info, batch_info)
        @timeit to "get_pre_relu_spec_A" batch_info = get_pre_relu_spec_A(batch_info[:spec_A_b], prop_method.use_gpu, true, model_info, batch_info)
    end
    if spec_bound_upper
        #batch_info = get_pre_relu_A(init, prop_method.use_gpu, false, model_info, batch_info)
        @timeit to "get_pre_relu_spec_A" batch_info = get_pre_relu_spec_A(batch_info[:spec_A_b], prop_method.use_gpu, false, model_info, batch_info)
    end
    return ConcretizeCrownBound(lower_spec_l, upper_spec_u, batch_bound.batch_data_min, batch_bound.batch_data_max), batch_info
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
            # println(batch_info[node][:pre_lower_A_function])
            # println(batch_info[node][:pre_upper_A_function])
            # @assert false
            A_function = use_gpu ? fmap(cu, Chain(batch_info[node][:pre_lower_A_function])) : Chain(batch_info[node][:pre_lower_A_function])
            batch_info[node][:pre_lower_spec_A] = A_function(init)[1]
            
            batch_info[node][:pre_upper_spec_A] = nothing
        end
    end
    if !lower_or_upper
        for node in model_info.activation_nodes
            A_function = use_gpu ? fmap(cu, Chain(batch_info[node][:pre_upper_A_function])) : Chain(batch_info[node][:pre_upper_A_function])
            batch_info[node][:pre_upper_spec_A] = A_function(init)[1]
            batch_info[node][:pre_lower_spec_A] = nothing
        end
    end
    return batch_info
end

"""
    check_inclusion(prop_method::BetaCrown, model, batch_input::AbstractArray, bound::ConcretizeCrownBound, batch_out_spec::LinearSpec)
"""
function check_inclusion(prop_method::BetaCrown, model, batch_input::AbstractArray, bound::ConcretizeCrownBound, batch_out_spec::LinearSpec)
    # spec_l, spec_u = process_bound(prop_method::AlphaCrown, bound, batch_out_spec, batch_info)
    batch_input = prop_method.use_gpu ? fmap(cu, batch_input) : batch_input
    spec_l, spec_u = bound.spec_l, bound.spec_u
    batch_size = length(batch_input)
    #center = (bound.batch_data_min[1:end,:] + bound.batch_data_max[1:end,:])./2 # out_dim x batch_size
    center = (bound.batch_data_min .+ bound.batch_data_max) ./ 2 # out_dim x batch_size
    model = prop_method.use_gpu ? fmap(cu, model) : model
    out_center = model(center)
    center_res = batched_vec(batch_out_spec.A, out_center) .- batch_out_spec.b # spec_dim x batch_size
    center_res = reshape(maximum(center_res, dims=1), batch_size) # batch_size
    results = [BasicResult(:unknown) for _ in 1:batch_size]

    # complement out spec: violated if exist y such that Ay-b < 0. Need to make sure lower bound of Ay-b > 0 to hold, spec_l > 0
    if batch_out_spec.is_complement
        @assert prop_method.bound_lower 
        spec_l = reshape(maximum(spec_l, dims=1), batch_size) # batch_size, min_x max_i of ai x - bi
        for i in 1:batch_size
            CUDA.@allowscalar center_res[i] <= 0 && (results[i] = BasicResult(:violated))
            CUDA.@allowscalar spec_l[i] > 0 && (results[i] = BasicResult(:holds))
        end
    else # polytope out spec: holds if all y such that Ay-b < 0. Need to make sure upper bound of Ay-b < 0 to hold.
        @assert prop_method.bound_upper
        spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
        for i in 1:batch_size
            CUDA.@allowscalar spec_u[i] <= 0 && (results[i] = BasicResult(:holds))
            CUDA.@allowscalar center_res[i] > 0 && (results[i] = BasicResult(:violated))
        end
    end
    return results
end 
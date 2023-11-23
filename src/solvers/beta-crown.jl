
mutable struct BetaCrown <: BatchBackwardProp 
    use_gpu::Bool
    split_neuron_number::Int
    pre_bound_method::Union{BatchForwardProp, BatchBackwardProp, Nothing, Dict}
    bound_lower::Bool
    bound_upper::Bool
    optimizer
    train_iteration::Int
end

struct BetaCrownBound <: Bound
    lower_A_x
    upper_A_x
    lower_A_W
    upper_A_W
    batch_data_min
    batch_data_max
end


function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::BetaCrown, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    model = prop_method.use_gpu ? fmap(cu, problem.Flux_model) : problem.Flux_model
    return model_info, Problem(problem.onnx_model_path, model, init_bound(prop_method, problem.input), problem.output)
end


function init_batch_bound(prop_method::BetaCrown, batch_input::AbstractArray, batch_output::LinearSpec)
    batch_data_min = prop_method.use_gpu ? fmap(cu, cat([low(h[1]) for h in batch_input]..., dims=2)) : cat([low(h[1]) for h in batch_input]..., dims=2)
    batch_data_max = prop_method.use_gpu ? fmap(cu, cat([high(h[1]) for h in batch_input]..., dims=2)) : cat([high(h[1]) for h in batch_input]..., dims=2)
    bound = BetaCrownBound([], [], nothing, nothing, batch_data_min, batch_data_max)
    return bound
end


function prepare_method(prop_method::BetaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    out_specs = get_linear_spec(batch_output)
    if prop_method.use_gpu
        out_specs = LinearSpec(fmap(cu, out_specs.A), fmap(cu, out_specs.b), fmap(cu, out_specs.is_complement))
    end
    return prepare_method(prop_method, batch_input, out_specs, model_info)
end

function prepare_method(prop_method::BetaCrown, batch_input::AbstractVector, out_specs::LinearSpec, model_info)
    #batch_input : (input, S_dict)
    batch_size = length(batch_input)
    if prop_method.use_gpu
        out_specs = LinearSpec(fmap(cu, out_specs.A), fmap(cu, out_specs.b), fmap(cu, out_specs.is_complement))
    end
    #prop_method.bound_lower = out_specs.is_complement ? true : false
    #prop_method.bound_upper = out_specs.is_complement ? false : true
    prop_method.bound_lower = true
    prop_method.bound_upper = true
    batch_info = init_propagation(prop_method, batch_input, out_specs, model_info)
    batch_info[:spec_A_b] = [out_specs.A, .-out_specs.b] # spec_A x < spec_b  ->  A x + b < 0, need negation
    batch_info[:init_upper_A_b] = [out_specs.A, .-out_specs.b]

    # for polytope output set, spec holds if upper bound of (spec_A x - b) < 0 for all dimension.
    # for complement polytope set, spec holds if lower bound of (spec_A x - b) > 0 for any dimension.

    # After conversion, we only need to decide if lower bound of spec_A y-spec_b > 0 or if upper bound of spec_A y - spec_b < 0
    # The new out is spec_A*y-b, whose dimension is spec_dim x batch_size.
    # Therefore, we set new_spec_A: 1(new_spec_dim) x original_spec_dim x batch_size, new_spec_b: 1(new_spec_dim) x batch_size,
    # spec_dim, out_dim, batch_size = size(out_specs.A)
    # out_specs = LinearSpec(ones((1, spec_dim, batch_size)), zeros(1, batch_size), out_specs.is_complement)

    # if hasproperty(prop_method, :pre_bound_method) && !isnothing(prop_method.pre_bound_method)
    #     Crown_input = [input[1] for input in batch_input]
    #     pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, Crown_input, batch_output, model_info)
    #     pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
    #     for node in model_info.activation_nodes
    #         @assert length(model_info.node_prevs[node]) == 1
    #         prev_node = model_info.node_prevs[node][1]
    #         batch_info[node][:pre_bound] = pre_batch_info[prev_node][:bound]
    #     end
    # end
    if prop_method.pre_bound_method isa Dict
        for node in model_info.activation_nodes
            batch_info[node][:pre_bound] = prop_method.pre_bound_method[node]
        end
    elseif prop_method.pre_bound_method isa BetaCrown  # requires recursive bounding, iterate from first layer
        # need forward BFS to compute pre_bound of all, 
        pre_bounds = Dict()

        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            
            sub_model_info = get_sub_model(model_info, prev_node)
            # println("=====")
            # println(sub_model_info.all_nodes)
            n_out = size(model_info.node_layer[prev_node].weight)[1]
            # println(prev_node)
            # println(n_out)
            # println(batch_size)

            I_spec = LinearSpec(repeat(Matrix(1.0I, n_out, n_out),1,1,batch_size), zeros(n_out, batch_size), false)
            if prop_method.use_gpu
                I_spec = LinearSpec(fmap(cu, I_spec.A), fmap(cu, I_spec.b), fmap(cu, I_spec.is_complement))
            end

            prop_method.pre_bound_method.pre_bound_method = pre_bounds

            sub_out_spec, sub_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, I_spec, sub_model_info)
            sub_batch_bound, sub_batch_info = propagate(prop_method.pre_bound_method, sub_model_info, sub_batch_info)
            sub_batch_bound, sub_batch_info = process_bound(prop_method.pre_bound_method, sub_batch_bound, sub_out_spec, sub_model_info, sub_batch_info)
            
            batch_info[node][:pre_bound] = sub_batch_bound
            pre_bounds[node] = sub_batch_bound

            # println(node)
            # println(prev_node)
            # println(sub_batch_info[prev_node][:bound])
        end
    elseif !isnothing(prop_method.pre_bound_method)
        pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, out_specs, model_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            batch_info[node][:pre_bound] = pre_batch_info[prev_node][:bound]
            # println("assigning", node," ", prev_node)
        end
    end
    
    batch_info[:batch_size] = length(batch_input)
    batch_info[:split_neuron_number] = prop_method.split_neuron_number
    for node in model_info.all_nodes
        batch_info[node][:beta] = 1
        batch_info[node][:max_split_number] = 1
        batch_info[node][:weight_ptb] = false
        batch_info[node][:bias_ptb] = false
    end
    #initialize alpha & beta
    for node in model_info.activation_nodes
        batch_info = init_alpha(model_info.node_layer[node], node, batch_info)
        batch_info = init_beta(model_info.node_layer[node], node, batch_info, batch_input)
    end
    n = size(out_specs.A, 2)
    batch_info[:init_A_b] = init_A_b(n, batch_info[:batch_size])
    batch_info[:Beta_Lower_Layer_node] = []#store the order of the node which has AlphaBetaLayer
    return out_specs, batch_info
end 


function init_A_b(n, batch_size) # A x < b
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    A = repeat(I, outer=(1, 1, batch_size))
    b = repeat(Z, outer=(1, 1, batch_size))
    return [A, b]
end

function init_bound(prop_method::BetaCrown, input) 
    return (input, Dict())
end

function process_bound(prop_method::BetaCrown, batch_bound::BetaCrownBound, batch_out_spec, model_info, batch_info)
    to = get_timer("Shared")
    @timeit to "compute_bound" compute_bound = Compute_bound(batch_bound.batch_data_min, batch_bound.batch_data_max)
    #bound_model = Chain(push!(prop_method.bound_lower ? batch_bound.lower_A_x : batch_bound.upper_A_x, compute_bound)) 
    bound_lower_model = Chain(push!(batch_bound.lower_A_x, compute_bound)) 
    bound_upper_model = Chain(push!(batch_bound.upper_A_x, compute_bound)) 
    bound_lower_model = prop_method.use_gpu ? fmap(cu, bound_lower_model) : bound_lower_model
    bound_upper_model = prop_method.use_gpu ? fmap(cu, bound_upper_model) : bound_upper_model
    # loss_func = prop_method.bound_lower ?  x -> - sum(x[1]) : x -> sum(x[2])

    # for polytope output set, spec holds if upper bound of (spec_A x - b) < 0 for all dimension. therefore minimize maximum(spec_A x - b)
    # for complement polytope set, spec holds if lower bound of (spec_A x - b) > 0 for any dimension. therefore maximize maximum(spec_A x - b), that is minimize -maximum(spec_A x - b)
    # loss_func = prop_method.bound_lower ?  x -> -maximum(x[1]) : x -> maximum(x[2]) # maximum leads to error in flux
    # loss_func = prop_method.bound_lower ?  x -> - sum(x[1]) : x -> sum(x[2])
    loss_func = prop_method.bound_lower ?  x -> -sum(exp.(x[1])) : x -> sum(exp.(x[2]))

    @timeit to "optimize_bound" bound_lower_model = optimize_bound(bound_lower_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
    @timeit to "optimize_bound" bound_upper_model = optimize_bound(bound_upper_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)

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
    # println(lower_spec_l)
    # println(upper_spec_u)
    # for polytope output set, spec holds if upper bound of (spec_A x - b) < 0 for all dimension.
    # for complement polytope set, spec holds if lower bound of (spec_A x - b) > 0 for any dimension.
    prop_method.bound_lower = batch_out_spec.is_complement ? true : false
    prop_method.bound_upper = batch_out_spec.is_complement ? false : true
    # println(prop_method.bound_lower)
    # println(prop_method.bound_upper)
    if prop_method.bound_lower
        #batch_info = get_pre_relu_A(init, prop_method.use_gpu, true, model_info, batch_info)
        @timeit to "get_pre_relu_spec_A" batch_info = get_pre_relu_spec_A(batch_info[:spec_A_b], prop_method.use_gpu, true, model_info, batch_info)
    end
    if prop_method.bound_upper
        #batch_info = get_pre_relu_A(init, prop_method.use_gpu, false, model_info, batch_info)
        @timeit to "get_pre_relu_spec_A" batch_info = get_pre_relu_spec_A(batch_info[:spec_A_b], prop_method.use_gpu, false, model_info, batch_info)
    end
    return ConcretizeCrownBound(lower_spec_l, upper_spec_u, batch_bound.batch_data_min, batch_bound.batch_data_max), batch_info
end


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
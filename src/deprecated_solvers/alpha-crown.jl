
mutable struct AlphaCrown <: BatchBackwardProp 
    use_gpu::Bool
    pre_bound_method::Union{BatchForwardProp, BatchBackwardProp, Nothing, Dict}
    bound_lower::Bool
    bound_upper::Bool
    optimizer
    train_iteration::Int
end

struct AlphaCrownBound <: Bound
    lower_A_x
    upper_A_x
    lower_A_W
    upper_A_W
    batch_data_min
    batch_data_max
end


function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::AlphaCrown, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    model = prop_method.use_gpu ? fmap(cu, problem.Flux_model) : problem.Flux_model
    return model_info, Problem(problem.onnx_model_path, model, init_bound(prop_method, problem.input), problem.output)
end

function prepare_method(prop_method::AlphaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    out_specs = get_linear_spec(batch_output)
    if prop_method.use_gpu
        out_specs = LinearSpec(fmap(cu, out_specs.A), fmap(cu, out_specs.b), fmap(cu, out_specs.is_complement))
    end
    return prepare_method(prop_method, batch_input, out_specs, model_info)
end

function prepare_method(prop_method::AlphaCrown, batch_input::AbstractVector, out_specs::LinearSpec, model_info)
    batch_size = size(out_specs.A, 3)
    prop_method.bound_lower = out_specs.is_complement ? true : false
    prop_method.bound_upper = out_specs.is_complement ? false : true
    batch_info = init_propagation(prop_method, batch_input, out_specs, model_info)
    
    batch_info[:spec_A_b] = [out_specs.A, .-out_specs.b] # spec_A x < spec_b  ->  A x + b < 0, need negation
    batch_info[:init_upper_A_b] = [out_specs.A, .-out_specs.b]

    # After conversion, we only need to decide if lower bound of spec_A y-spec_b > 0 or if upper bound of spec_A y - spec_b < 0
    # The new out is spec_A*y-b, whose dimension is spec_dim x batch_size.
    # Therefore, we set new_spec_A: 1(new_spec_dim) x original_spec_dim x batch_size, new_spec_b: 1(new_spec_dim) x batch_size,
    # spec_dim, out_dim, batch_size = size(out_specs.A)
    # out_specs = LinearSpec(ones((1, spec_dim, batch_size)), zeros(1, batch_size), out_specs.is_complement)
    if prop_method.pre_bound_method isa Dict
        for node in model_info.activation_nodes
            batch_info[node][:pre_bound] = prop_method.pre_bound_method[node]
        end
    elseif prop_method.pre_bound_method isa AlphaCrown  # requires recursive bounding, iterate from first layer
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
    
    #initialize alpha 
    for node in model_info.activation_nodes
        batch_info = init_alpha(model_info.node_layer[node], node, batch_info, batch_input)
    end

    for node in model_info.all_nodes
        batch_info[node][:beta] = 1
        batch_info[node][:weight_ptb] = false
        batch_info[node][:bias_ptb] = false
    end
    
    batch_info[:Alpha_Lower_Layer_node] = []#store the order of the node which has AlphaLayer
    batch_info[:batch_size] = length(batch_input)
    # init_A_b(prop_method, batch_input, batch_info)
    return out_specs, batch_info
end


function init_batch_bound(prop_method::AlphaCrown, batch_input::AbstractArray, batch_output::LinearSpec)
    batch_data_min = prop_method.use_gpu ? fmap(cu, cat([low(h) for h in batch_input]..., dims=2)) : cat([low(h) for h in batch_input]..., dims=2)
    batch_data_max = prop_method.use_gpu ? fmap(cu, cat([high(h) for h in batch_input]..., dims=2)) : cat([high(h) for h in batch_input]..., dims=2)
    bound = AlphaCrownBound([], [], nothing, nothing, batch_data_min, batch_data_max)
    return bound
end



function process_bound(prop_method::AlphaCrown, batch_bound::AlphaCrownBound, batch_out_spec, model_info, batch_info)
    #println("batch_bound.batch_data_min max")
    #println(size(batch_bound.batch_data_min), size(batch_bound.batch_data_max))
    compute_bound = Compute_bound(batch_bound.batch_data_min, batch_bound.batch_data_max)
    bound_model = Chain(push!(prop_method.bound_lower ? batch_bound.lower_A_x : batch_bound.upper_A_x, compute_bound)) 
    bound_model = prop_method.use_gpu ? fmap(cu, bound_model) : bound_model
    # maximize lower(A * x - b) or minimize upper(A * x - b)
    loss_func = prop_method.bound_lower ?  x -> - sum(x[1]) : x -> sum(x[2])

    bound_model = optimize_model(bound_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
    
    for (index, params) in enumerate(Flux.params(bound_model))
        relu_node = batch_info[:Alpha_Lower_Layer_node][index]
        batch_info[relu_node][prop_method.bound_lower ? :alpha_lower : :alpha_upper] = params
        #println(relu_node)
        #println(params)
    end
    spec_l, spec_u = bound_model(batch_info[:spec_A_b])
    # println("spec_l, spec_u")
    # println(spec_l)
    # println(spec_u)
    return ConcretizeCrownBound(spec_l, spec_u, batch_bound.batch_data_min, batch_bound.batch_data_max), batch_info
end

function check_inclusion(prop_method::AlphaCrown, model, batch_input::AbstractArray, bound::ConcretizeCrownBound, batch_out_spec::LinearSpec)
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
        if prop_method.use_gpu
            for i in 1:batch_size
                CUDA.@allowscalar center_res[i] <= 0 && (results[i] = BasicResult(:violated))
                CUDA.@allowscalar spec_l[i] > 0 && (results[i] = BasicResult(:holds))
            end
        else
            for i in 1:batch_size
                center_res[i] <= 0 && (results[i] = BasicResult(:violated))
                spec_l[i] > 0 && (results[i] = BasicResult(:holds))
            end
        end
    else # polytope out spec: holds if all y such that Ay-b < 0. Need to make sure upper bound of Ay-b < 0 to hold.
        @assert prop_method.bound_upper
        spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
        if prop_method.use_gpu
            for i in 1:batch_size
                CUDA.@allowscalar spec_u[i] <= 0 && (results[i] = BasicResult(:holds))
                CUDA.@allowscalar center_res[i] > 0 && (results[i] = BasicResult(:violated))
            end
        else
            for i in 1:batch_size
                CUDA.@allowscalar spec_u[i] <= 0 && (results[i] = BasicResult(:holds))
                CUDA.@allowscalar center_res[i] > 0 && (results[i] = BasicResult(:violated))
            end
        end
    end
    return results
end


mutable struct AlphaLayer
    node
    alpha
    lower
    unstable_mask
    active_mask 
    upper_slope
    lower_bias
    upper_bias
end
Flux.@functor AlphaLayer (alpha,) #only alpha need to be trained


function (f::AlphaLayer)(x)
    last_A = x[1]
    if isnothing(last_A)
        return [New_A, nothing]
    end

    lower_slope = clamp.(f.alpha, 0, 1) .* f.unstable_mask .+ f.active_mask 
    if f.lower 
        New_A = bound_oneside(last_A, lower_slope, f.upper_slope)
    else
        New_A = bound_oneside(last_A, f.upper_slope, lower_slope)
    end

    if f.lower 
        New_bias = multiply_bias(last_A, f.lower_bias, f.upper_bias) .+ x[2]
    else
        New_bias = multiply_bias(last_A, f.upper_bias, f.lower_bias) .+ x[2]
    end
    return [New_A, New_bias]
end

function propagate_act_batch(prop_method::AlphaCrown, layer::typeof(relu), bound::AlphaCrownBound, batch_info)
    node = batch_info[:current_node]
   #=  if !haskey(batch_info[node], :pre_lower) || !haskey(batch_info[node], :pre_upper)
        lower, upper = compute_bound(batch_info[node][:pre_bound])
        batch_info[node][:pre_lower] = lower
        batch_info[node][:pre_upper] = upper
    else =#
    lower = batch_info[node][:pre_lower]  
    upper = batch_info[node][:pre_upper]
    #end

    alpha_lower = batch_info[node][:alpha_lower]
    alpha_upper = batch_info[node][:alpha_upper]
    upper_slope, upper_bias = relu_upper_bound(lower, upper) #upper_slope:upper of slope  upper_bias:Upper of bias
    lower_bias = prop_method.use_gpu ? fmap(cu, zeros(size(upper_bias))) : zeros(size(upper_bias))
    active_mask = (lower .>= 0)
    inactive_mask = (upper .<= 0)
    unstable_mask = (upper .> 0) .& (lower .< 0)
    batch_info[node][:unstable_mask] = unstable_mask
    
    lower_A = bound.lower_A_x
    upper_A = bound.upper_A_x
    
    batch_info[node][:pre_lower_A_function] = nothing
    batch_info[node][:pre_upper_A_function] = nothing

    if prop_method.bound_lower
        batch_info[node][:pre_lower_A_function] = copy(lower_A)
        Alpha_Lower_Layer = AlphaLayer(node, alpha_lower, true, unstable_mask, active_mask, upper_slope, lower_bias, upper_bias)
        push!(lower_A, Alpha_Lower_Layer)
    end

    if prop_method.bound_upper
        batch_info[node][:pre_upper_A_function] = copy(upper_A)
        Alpha_Upper_Layer = AlphaLayer(node, alpha_upper, false, unstable_mask, active_mask, upper_slope, lower_bias, upper_bias)
        push!(upper_A, Alpha_Upper_Layer)
    end
    push!(batch_info[:Alpha_Lower_Layer_node], node)
    New_bound = AlphaCrownBound(lower_A, upper_A, nothing, nothing, bound.batch_data_min, bound.batch_data_max)
    return New_bound
end


function propagate_linear_batch(prop_method::AlphaCrown, layer::Dense, bound::AlphaCrownBound, batch_info)
    node = batch_info[:current_node]
    #TO DO: we haven't consider the perturbation in weight and bias
    bias_lb = _preprocess(node, batch_info, layer.bias)
    bias_ub = _preprocess(node, batch_info, layer.bias)
    lA_W = uA_W = lA_bias = uA_bias = lA_x = uA_x = nothing
    if !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
        weight = layer.weight
        bias = bias_lb
        if prop_method.bound_lower
            lA_x = dense_bound_oneside(bound.lower_A_x, weight, bias, batch_info[:batch_size])
        else
            lA_x = nothing
        end
        if prop_method.bound_upper
            uA_x = dense_bound_oneside(bound.upper_A_x, weight, bias, batch_info[:batch_size])
        else
            uA_x = nothing
        end
        New_bound = AlphaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max)
        return New_bound
    end
end
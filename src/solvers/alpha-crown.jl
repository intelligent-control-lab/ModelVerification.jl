
mutable struct AlphaCrown <: BatchBackwardProp 
    use_gpu::Bool
    pre_bound_method::Union{BatchForwardProp, BatchBackwardProp, Nothing}
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


function prepare_method(prop_method::AlphaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    out_specs = get_linear_spec(batch_output)
    if prop_method.use_gpu
        out_specs = LinearSpec(fmap(cu, out_specs.A), fmap(cu, out_specs.b), fmap(cu, out_specs.is_complement))
    end
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

    if hasproperty(prop_method, :pre_bound_method) && !isnothing(prop_method.pre_bound_method)
        pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, batch_output, model_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            batch_info[node][:pre_bound] = pre_batch_info[prev_node][:bound]
        end
    end
    
    #initialize alpha 
    for node in model_info.activation_nodes
        batch_info = init_alpha(model_info.node_layer[node], node, batch_info)
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


function process_bound(prop_method::AlphaCrown, batch_bound::AlphaCrownBound, batch_out_spec, model_info, batch_info)
    #println("batch_bound.batch_data_min max")
    #println(size(batch_bound.batch_data_min), size(batch_bound.batch_data_max))
    compute_bound = Compute_bound(batch_bound.batch_data_min, batch_bound.batch_data_max)
    bound_model = Chain(push!(prop_method.bound_lower ? batch_bound.lower_A_x : batch_bound.upper_A_x, compute_bound)) 
    bound_model = prop_method.use_gpu ? fmap(cu, bound_model) : bound_model
    # maximize lower(A * x - b) or minimize upper(A * x - b)
    loss_func = prop_method.bound_lower ?  x -> - sum(x[1]) : x -> sum(x[2])

    bound_model = optimize_bound(bound_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
    
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

function optimize_bound(model, input, loss_func, optimizer, max_iter)
    min_loss = Inf
    opt_state = Flux.setup(optimizer, model)
    for i in 1 : max_iter
        losses, grads = Flux.withgradient(model) do m
            result = m(input) 
            loss_func(result)
        end
        if losses <= min_loss
            min_loss = min_loss
        else
            return model
        end
        Flux.update!(opt_state, model, grads[1])
    end
    return model
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
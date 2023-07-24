@with_kw struct BFS <: SearchMethod
    max_iter::Int64
    batch_size::Int64 = 1
end

function search_branches(search_method::BFS, split_method, prop_method, problem, model_info)
    branches = [(problem.input, problem.output)]
    batch_input = []
    batch_output = []
    for iter in 1:search_method.max_iter # BFS with max iteration
        length(branches) == 0 && break
        input, output = popfirst!(branches)
        push!(batch_input, input)
        push!(batch_output, output)
        if length(batch_input) >= search_method.batch_size || length(branches) == 0
            batch_info = init_start_node_bound(prop_method, batch_input, model_info)
            batch_out_spec, batch_info = prepare_method(prop_method, batch_input, batch_output, model_info, batch_info)
            batch_bound, batch_info = propagate(prop_method, model_info, batch_out_spec, batch_info)
            batch_bound, batch_info = process_bound(prop_method, batch_bound, batch_out_spec, batch_info)
            batch_result = check_inclusion(prop_method, problem.Flux_model, batch_input, batch_bound, batch_out_spec)
            for i in eachindex(batch_input)
                batch_result[i].status == :holds && continue
                batch_result[i].status == :violated && return batch_result[i]
                # batch_result[i].status == :unknown
                sub_branches = split_branch(split_method, problem.Flux_model, batch_input[i], batch_output[i])
                branches = [branches; sub_branches]
            end
            batch_input = []
            batch_output = []
        end
    end
    length(branches) == 0 && return BasicResult(:holds)
    return BasicResult(:unknown)
end


struct Compute_bound
    batch_data_min
    batch_data_max
    spec_A
    spec_b
end
Flux.@functor Compute_bound ()

function (f::Compute_bound)(x)
    z = zeros(size(x[1]))
    l = batched_mul(max.(x[1], z), f.batch_data_min) .+ batched_mul(min.(x[1], z), f.batch_data_max) .+ x[2]
    u = batched_mul(max.(x[1], z), f.batch_data_max) + batched_mul(min.(x[1], z), f.batch_data_min) .+ x[2]
    print("l")
    println(l)
    print("u")
    println(u) 
    pos_A = max.(f.spec_A, zeros(size(f.spec_A))) # spec_dim x out_dim x batch_size
    neg_A = min.(f.spec_A, zeros(size(f.spec_A)))
    spec_u = batched_mul(pos_A, u) + batched_mul(neg_A, l) .- f.spec_b # spec_dim x batch_size
    spec_l = batched_mul(pos_A, l) + batched_mul(neg_A, u) .- f.spec_b # spec_dim x batch_size

    return [spec_l, spec_u]
end 

function process_bound(prop_method::PropMethod, batch_bound, batch_out_spec, batch_info)
    return batch_bound, batch_info
end

function process_bound(prop_method::AlphaCrown, batch_bound, batch_out_spec, batch_info)
    compute_bound = Compute_bound(batch_bound.batch_data_min, batch_bound.batch_data_max, batch_out_spec.A, batch_out_spec.b)
    final_spec_l = final_spec_u = nothing
    if  prop_method.bound_lower#batch_out_spec.is_complement = true
        input_lower_A_bias = batch_info[:init_lower_A_bias] #batch_info[:init_lower_A_bias] stores the input lower A(Identity Matrix)
        optimize_lower_A_function = batch_bound.lower_A_x
        push!(optimize_lower_A_function, compute_bound)
        optimize_lower_A_model = Chain(optimize_lower_A_function)
        state_lower_function = Flux.setup(prop_method.optimizer, optimize_lower_A_model)
        loss_lower(x) = 0.0 - sum(x) #lower(A * x - b) > 0
        for i in 1 : prop_method.trian_iteration
            lower_loss, lower_grads = Flux.withgradient(optimize_lower_A_model) do m
            result = m(input_lower_A_bias)[1] #spec_l
            loss_lower(result)
            end
            Flux.Optimise.update!(state_lower_function, optimize_lower_A_model, lower_grads[1])
        end

        for (index, params) in enumerate(Flux.params(optimize_lower_A_model))
            relu_node = batch_info[:Alpha_Lower_Layer_node][index]
            batch_info[relu_node][:alpha_lower] = params
        end

        final_spec_l = optimize_lower_A_model(input_lower_A_bias)[1]
        final_spec_u = optimize_lower_A_model(input_lower_A_bias)[2]
        #result_bound = AlphaCrownBound(final_result[1], final_result[2], nothing, nothing, batch_bound.batch_data_min, batch_bound.batch_data_max)
    end

    if  prop_method.bound_upper#batch_out_spec.is_complement = false
        input_upper_A_bias = batch_info[:init_upper_A_bias] #batch_info[:init_upper_A_bias] stores the input upper A(Identity Matrix)
        optimize_upper_A_function = batch_bound.upper_A_x
        push!(optimize_upper_A_function, compute_bound)
        optimize_upper_A_model = Chain(optimize_upper_A_function)
        state_uppper_function = Flux.setup(prop_method.optimizer, optimize_upper_A_model)
        loss_upper(x) = sum(x) - 0.0 #upper(A * x - b) < 0
        for i in 1 : prop_method.trian_iteration
            upper_loss, upper_grads = Flux.withgradient(optimize_upper_A_model) do m
            result = m(input_upper_A_bias)[2] #spec_u
            loss_upper(result)
            end
            Flux.Optimise.update!(state_uppper_function, optimize_upper_A_model, upper_grads[1])
        end

        for (index, params) in enumerate(Flux.params(optimize_upper_A_model))
            relu_node = batch_info[:Alpha_Lower_Layer_node][index]
            batch_info[relu_node][:alpha_upper] = params
        end
        
        final_spec_l = optimize_upper_A_model(input_upper_A_bias)[1]
        final_spec_u = optimize_upper_A_model(input_upper_A_bias)[2]
        #result_bound = AlphaCrownBound(final_result[1], final_result[2], nothing, nothing, batch_bound.batch_data_min, batch_bound.batch_data_max)
    end

    result_bound = AlphaCrownBound(final_spec_l, final_spec_u, nothing, nothing, batch_bound.batch_data_min, batch_bound.batch_data_max)
    return result_bound, batch_info
end
const DEFAULT_PRE_ACT_BOUND::FloatType[] = 1e4

@with_kw struct MIPVerify <: ForwardProp
    optimizer = GLPK.Optimizer
    pre_bound_method::Union{ForwardProp, BackwardProp, Nothing} = nothing
end

function prepare_method(
    prop_method::MIPVerify,
    batch_input::AbstractVector,
    batch_output::AbstractVector,
    batch_inheritance::AbstractVector,
    model_info::ModelGraph,
)::Tuple{AbstractVector, Dict}
    batch_info = init_propagation(prop_method, batch_input, batch_output, model_info)
    
    # store :size_after_layer
    batch_info = get_all_layer_output_size(model_info, batch_info, size(LazySets.center(batch_input[1])))

    # store previous nodes
    for node in model_info.all_nodes
        batch_info[node][:prev_nodes] = model_info.node_prevs[node]
    end

    # compute pre-activation bounds
    if !isnothing(prop_method.pre_bound_method)
        # compute bounds using specified method
        _, pre_batch_info = prepare_method(
            prop_method.pre_bound_method, batch_input, batch_output, batch_inheritance, model_info)
        _, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            pre_bound = pre_batch_info[prev_node][:bound]
            l, u = compute_bound(pre_bound)
            # convert CUDA.CuArray to Vector for indexing in relu propagation
            batch_info[node][:pre_lower] = collect(l[:, 1])
            batch_info[node][:pre_upper] = collect(u[:, 1])
        end
    else
        # initialize bounds as sufficiently large intervals
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            prev_size = batch_info[prev_node][:size_after_layer][1]
            batch_info[node][:pre_lower] = fill(-DEFAULT_PRE_ACT_BOUND, prev_size)
            batch_info[node][:pre_upper] = fill(DEFAULT_PRE_ACT_BOUND, prev_size)
        end
    end

    # create optimization model
    opt_model = Model(prop_method.optimizer)
    batch_info[:opt_model] = opt_model

    # create input variables and add constraints (suppose there is only one input)
    start_node = model_info.start_nodes[1]
    z = @variable(opt_model, [1:batch_info[start_node][:size_after_layer][1]])
    add_set_constraint(opt_model, z, batch_input[1])
    batch_info[start_node][:opt_vars] = Dict(:z => z)

    # add objective function (suppose there is only one input)
    disturbance = z - LazySets.center(batch_input[1])
    obj = symbolic_infty_norm(opt_model, disturbance)
    @objective(opt_model, Min, obj)
    batch_info[:objective] = obj

    return batch_output, batch_info
end

function process_bound(
    prop_method::MIPVerify,
    batch_bound::AbstractArray,
    batch_out_spec::AbstractArray,
    model_info::ModelGraph,
    batch_info::Dict,
)::Tuple{Vector{Hyperrectangle}, Dict}
    opt_model = batch_info[:opt_model]
    final_node = model_info.final_nodes[1]
    z = batch_info[final_node][:opt_vars][:z]

    function get_max_disturbance()
        if termination_status(opt_model) == OPTIMAL
            max_disturbance = value(batch_info[:objective])
            # counterexample is available but is nowhere to return
            counterexample = value.(batch_info[model_info.start_nodes[1]][:opt_vars][:z])
        else
            max_disturbance = Inf
        end
        return max_disturbance
    end

    # add output constraints (suppose there is only one output)
    if batch_out_spec[1] isa Complement
        # constraint is convex, solve one optimization problem
        add_set_constraint(opt_model, z, batch_out_spec[1].X)
        optimize!(opt_model)
        max_disturbance = get_max_disturbance()
    else
        # constraint is not convex, split into `n` Hyperplanes and solve `n` optimization problems
        A, b = tosimplehrep(batch_out_spec[1])
        max_disturbance = Inf
        output_constraint = nothing
        for i in axes(A, 1)
            if i > 1
                JuMP.delete(opt_model, output_constraint)
                JuMP.unregister(opt_model, :output_constraint)
            end
            @constraint(opt_model, output_constraint, dot(A[i, :], z) >= b[i])
            optimize!(opt_model)
            max_disturbance = min(get_max_disturbance(), max_disturbance)
            # property is violated as long as one counterexample is found
            if max_disturbance != Inf
                break
            end
        end
    end
    # ensure disturbance is non-negative in case of numerical errors
    max_disturbance = max(max_disturbance, 0.0)

    # wrap maximum disturbance as a Hyperrectangle for type consistency
    batch_bound = [Hyperrectangle([0.0], [max_disturbance])]

    return batch_bound, batch_info
end

function check_inclusion(
    prop_method::MIPVerify,
    model::Chain,
    batch_input::AbstractArray,
    batch_bound::Vector{Hyperrectangle},
    batch_out_spec::AbstractArray,
)::Vector{AdversarialResult}
    # obtain result according to maximum disturbance
    max_disturbance = radius(batch_bound[1])
    if max_disturbance == Inf
        result = AdversarialResult(:holds)
    else
        result = AdversarialResult(:violated, max_disturbance)
    end
    return [result]
end

function symbolic_infty_norm(opt_model::Model, var::AbstractVector)
    aux = @variable(opt_model)
    @constraint(opt_model, aux .>= var)
    @constraint(opt_model, aux .>= -var)
    return aux
end

function add_set_constraint(opt_model::Model, var::AbstractVector, set::LazySet)
    A, b = tosimplehrep(set)
    @constraint(opt_model, A * var .<= b)
end

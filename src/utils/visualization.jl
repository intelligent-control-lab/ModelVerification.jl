using Plots
using JLD2
using FileIO

function compute_all_bound(prop_method::ForwardProp, batch_input, model_info, out_and_bounds)

    batch_info = init_propagation(prop_method, batch_input, nothing, model_info)
    _, all_bounds = propagate(prop_method, model_info, batch_info)

    for node in model_info.all_nodes
        # @show node
        # @show haskey(all_bounds[node], :bound)
        haskey(all_bounds[node], :bound) || continue
        
        batch_out = out_and_bounds[node][:out]
        all_bounds[node][:out] = batch_out

        batch_bound = all_bounds[node][:bound]
        
        if typeof(batch_bound) <: Vector
            @assert length(batch_bound) == 1
            stats = @timed l, u = compute_bound(batch_bound[1])
        else
            stats = @timed l, u = compute_bound(batch_bound)
        end
        # println("bound time:", stats.time) 
        if (ndims(batch_out) == 4) && (ndims(l) == 2)
            l = reshape(l, (size(batch_out)[1:3]..., size(l)[2]))
            u = reshape(u, (size(batch_out)[1:3]..., size(u)[2]))
        end
        all_bounds[node][:l] = l
        all_bounds[node][:u] = u
        # @show node
        # @show l
        # @show u
    end
    return all_bounds
end

function compute_all_bound(prop_method::BackwardProp, batch_input::AbstractVector, model_info, nominal_outputs)
    
    batch_size = length(batch_input)
    
    all_bounds = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)

    for node in model_info.all_nodes
        if node in model_info.start_nodes
            continue
        end
        
        # @show node

        sub_model_info = get_sub_model(model_info, node)

        # need to revise the following for image inputs with Convolution layers
        n_out = size(nominal_outputs[node][:out])[1]
        I_spec = LinearSpec(repeat(Matrix(1.0I, n_out, n_out),1,1,batch_size), zeros(n_out, batch_size), false)
        if prop_method.use_gpu
            I_spec = LinearSpec(fmap(cu, I_spec.A), fmap(cu, I_spec.b), fmap(cu, I_spec.is_complement))
        end
        
        sub_out_spec, sub_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, I_spec, [all_bounds], sub_model_info)
        sub_batch_bound, sub_batch_info = propagate(prop_method.pre_bound_method, sub_model_info, sub_batch_info)
        sub_batch_bound, sub_batch_info = process_bound(prop_method.pre_bound_method, sub_batch_bound, sub_out_spec, sub_model_info, sub_batch_info)
        l, u = compute_bound(sub_batch_bound) # reach_dim x batch 
        # @show node
        # @show l
        # @show u
        all_bounds[node][:l] = l
        all_bounds[node][:u] = u
        for next_node in model_info.node_nexts[node]
            if next_node in model_info.activation_nodes
                all_bounds[next_node][:pre_lower] = l
                all_bounds[next_node][:pre_upper] = u
            end
        end
        # @show all_bounds
        # @show all_bounds
    end
    return all_bounds
end

function plot_bounds(all_bounds, model_info, save_path; vis_center=true, save_bound=false, plot_mode=:cr)

    if !isnothing(save_path)
        dir = dirname(save_path)
        if !isdir(dir)
            mkdir(dir)
        end
    end

    SNRs = []
    i = 0
    for node in model_info.all_nodes
        if node in model_info.start_nodes
            continue
        end
        if !isnothing(save_path) && !isnothing(all_bounds[node])
            i += 1
            # @show node
            
            l, u = all_bounds[node][:l], all_bounds[node][:u]

            println("saving visualized bound: ", save_path * string(i) * "_" * node * ".png")
            # @show size(l)
            # @show size(u)
            
            out_center = ndims(l) == 4 ? (u + l)[:,:,1,1]./2 : reshape((u + l)./2, :,1)
            out_radius = ndims(l) == 4 ? (u - l)[:,:,1,1] : reshape(u - l, :,1)
            
            out_l = ndims(l) == 4 ? l[:,:,1,1] : reshape(l, :,1)
            out_u = ndims(u) == 4 ? u[:,:,1,1] : reshape(u, :,1)
            
            # @show size(out_center)
            # @show plot_mode
            if plot_mode == :lu
                global_min = minimum(out_l)
                global_max = maximum(out_u)
                p1 = heatmap(out_l, clims=(global_min, global_max), c = :ice)            
                title!(string(i) * "_" * node * "_lower_bound")
                p2 = heatmap(out_u, clims=(global_min, global_max), c = :ice)
                title!(string(i) * "_" * node * "_upper_bound")
                plot(p1, p2, layout=(1,2), size = (800,300))
                savefig(save_path * string(i) * "_" * node * ".png")
            elseif plot_mode == :cr
                p1 = heatmap(out_center)            
                title!(string(i) * "_" * node * "_center")
                p2 = heatmap(out_radius, c = :ice)
                title!(string(i) * "_" * node * "_bound_size")
                plot(p1, p2, layout=(1,2), size = (800,300))
                savefig(save_path * string(i) * "_" * node * ".png")
                # plot(p2, size = (400,300))
                # savefig(save_path * string(i) * "_" * node * ".png")
            end

            signal_scale = maximum(out_center) - minimum(out_center)
            noise_scale = maximum(out_radius) - minimum(out_radius)
            push!(SNRs, signal_scale / noise_scale)

            if save_bound
                path = save_path * string(i) * "_" * node * "_info.jld2"
                bound = batch_info[node]
                @save path bound
            end
        end
    end
end

function visualize(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem, save_path; 
                    vis_center=true, 
                    save_bound=false,
                    plot_mode=:lu
                    )
    
    model_info, processed_problem = prepare_problem(search_method, split_method, prop_method, problem)
    processed_batch_input = [processed_problem.input]

    sample = center(problem.input)
    original_batch_input = reshape(sample, (size(sample)..., 1))
    nominal_outputs = compute_output(model_info, original_batch_input) # useful for infer output shapes

    all_bounds = compute_all_bound(prop_method, processed_batch_input, model_info, nominal_outputs)

    plot_bounds(all_bounds, model_info, save_path; vis_center=true, save_bound=false, plot_mode=plot_mode)
    return all_bounds
end

function center(bound::LazySet)
    return LazySets.center(bound)
end
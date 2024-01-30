using Plots
using JLD2
using FileIO

"""
    propagate_once(prop_method::PropMethod, model_info, batch_info, 
                   save_path;  vis_center=true, save_bound=false)

                   
"""
function propagate_once(prop_method::PropMethod, model_info, batch_info, save_path; vis_center=true, save_bound=false)
    if !isnothing(save_path)
        dir = dirname(save_path)
        if !isdir(dir)
            mkdir(dir)
        end
    end

    queue = Queue{Any}()
    enqueue_nodes!(prop_method, queue, model_info)
    out_cnt = Dict(node => 0 for node in model_info.all_nodes)
    visit_cnt = Dict(node => 0 for node in model_info.all_nodes)
    i = 0

    SNRs = []
    out_and_bounds = compute_output(model_info, batch_info)

    while !isempty(queue)
        i += 1
        
        node = dequeue!(queue)
        batch_info[:current_node] = node
        
        for output_node in next_nodes(prop_method, model_info, node)
            visit_cnt[output_node] += 1
            if all_prevs_in(prop_method, model_info, output_node, visit_cnt[output_node])
                enqueue!(queue, output_node)
            end
        end

        if has_two_reach_node(prop_method, model_info, node)
            batch_bound = propagate_skip_method(prop_method, model_info, batch_info, node)
        else
            batch_bound = propagate_layer_method(prop_method, model_info, batch_info, node)
        end

        batch_out = out_and_bounds[node][:out]

        batch_info[node][:bound] = batch_bound
        batch_info[node][:out] = batch_out
        
        if !isnothing(save_path)
            println("saving visualized bound: ", save_path * string(i) * "_" * node * ".png")
            if typeof(batch_bound) <: Vector
                @assert length(batch_bound) == 1
                stats = @timed l, u = compute_bound(batch_bound[1])
            else
                stats = @timed l, u = compute_bound(batch_bound)
            end
            println("bound time:", stats.time) 
            if (ndims(batch_out) == 4) && (ndims(l) == 2)
                l = reshape(l, (size(batch_out)[1:3]..., size(l)[2]))
                u = reshape(u, (size(batch_out)[1:3]..., size(u)[2]))
            end
            batch_info[node][:l] = l
            batch_info[node][:u] = u

            @show size(l)
            @show size(u)
            
            img = ndims(batch_out) == 4 ? batch_out[:,:,1,1] : reshape(batch_out, :,1)
            # img = ndims(l) == 4 ? (u + l)[:,:,1,1]./2 : reshape((u + l)./2, :,1)
            @show size(img)

            p_center = heatmap(img)
            title!(string(i) * "_" * node * "_clean")
            # title!(string(i) * "_" * node * "_center")

            signal_scale = maximum(img) - minimum(img)

            img = ndims(l) == 4 ? (u - l)[:,:,1,1] : reshape(u - l, :,1)
            p_lu = heatmap(img, c = :ice)
            title!(string(i) * "_" * node * "_bound_size")

            noise_scale = maximum(img) - minimum(img)
            
            push!(SNRs, signal_scale / noise_scale)

            if save_bound
                path = save_path * string(i) * "_" * node * "_info.jld2"
                bound = batch_info[node]
                @save path bound
            end
            if vis_center
                plot(p_center, p_lu, layout=(1,2), size = (800,300))
                savefig(save_path * string(i) * "_" * node * ".png")
            else
                plot(p_lu, size = (400,300))
                savefig(save_path * string(i) * "_" * node * ".png")
            end
        end
        
    end
    if !isnothing(save_path)
        plot(SNRs, xlabel = "layer", ylabel="SNR", legend=false, yaxis=:log)
        savefig(save_path * "_SNRs_log.png")
    end

    return batch_info
end

function backward_compute_bound(prop_method::BetaCrown, batch_input::AbstractVector, model_info, nominal_outputs)
    batch_size = length(batch_input)
    
    all_bounds = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)

    for node in model_info.all_nodes
        if node in model_info.start_nodes
            continue
        end
        
        @show node

        sub_model_info = get_sub_model(model_info, node)

        n_out = size(nominal_outputs[node][:out])[1]
        
        # need to revise the following for image inputs with Convolution layers
        I_spec = LinearSpec(repeat(Matrix(1.0I, n_out, n_out),1,1,batch_size), zeros(n_out, batch_size), false)
        if prop_method.use_gpu
            I_spec = LinearSpec(fmap(cu, I_spec.A), fmap(cu, I_spec.b), fmap(cu, I_spec.is_complement))
        end
        
        sub_out_spec, sub_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, I_spec, [all_bounds], sub_model_info)
        sub_batch_bound, sub_batch_info = propagate(prop_method.pre_bound_method, sub_model_info, sub_batch_info)
        sub_batch_bound, sub_batch_info = process_bound(prop_method.pre_bound_method, sub_batch_bound, sub_out_spec, sub_model_info, sub_batch_info)
        l, u = compute_bound(sub_batch_bound) # reach_dim x batch 
        all_bounds[node][:l] = l
        all_bounds[node][:u] = u
        for next_node in model_info.node_nexts[node]
            if next_node in model_info.activation_nodes
                all_bounds[next_node][:pre_lower] = l
                all_bounds[next_node][:pre_upper] = u
            end
        end
        # @show all_bounds
    end
    return all_bounds
end

function plot_bounds(all_bounds, model_info, save_path; vis_center=true, save_bound=false)
    SNRs = []
    for (i,node) in enumerate(model_info.all_nodes)
        if node in model_info.start_nodes
            continue
        end
        if !isnothing(save_path) && !isnothing(all_bounds[node])
            @show node
            
            l, u = all_bounds[node][:l], all_bounds[node][:u]

            println("saving visualized bound: ", save_path * string(i) * "_" * node * ".png")
            @show size(l)
            @show size(u)
            img = ndims(l) == 4 ? (u + l)[:,:,1,1]./2 : reshape((u + l)./2, :,1)
            @show size(img)
            p_center = heatmap(img)
            
            title!(string(i) * "_" * node * "_center")

            signal_scale = maximum(img) - minimum(img)

            img = ndims(l) == 4 ? (u - l)[:,:,1,1] : reshape(u - l, :,1)
            p_lu = heatmap(img, c = :ice)
            title!(string(i) * "_" * node * "_bound_size")

            noise_scale = maximum(img) - minimum(img)
            
            push!(SNRs, signal_scale / noise_scale)

            if save_bound
                path = save_path * string(i) * "_" * node * "_info.jld2"
                bound = batch_info[node]
                @save path bound
            end
            if vis_center
                plot(p_center, p_lu, layout=(1,2), size = (800,300))
                savefig(save_path * string(i) * "_" * node * ".png")
            else
                plot(p_lu, size = (400,300))
                savefig(save_path * string(i) * "_" * node * ".png")
            end
        end
    end
end
function visualize(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem, save_path; vis_center=true, save_bound=false)
    sample = center(problem.input)
    batch_input = reshape(sample, (size(sample)..., 1))
    @show size(batch_input)
    model_info, problem = prepare_problem(search_method, split_method, prop_method, problem)
    nominal_outputs = compute_output(model_info, batch_input)

    batch_out_spec, batch_info = prepare_method(prop_method, [problem.input], [problem.output], [nothing], model_info)

    if typeof(prop_method) <: ForwardProp
        batch_info = propagate_once(prop_method, model_info, batch_info, save_path; vis_center=vis_center, save_bound=save_bound)
        batch_bound = batch_info[output_node(prop_method, model_info)][:bound]
        batch_result = check_inclusion(prop_method, problem.Flux_model, [problem.input], batch_bound, batch_out_spec)
    elseif typeof(prop_method) <: BackwardProp
        batch_bound, batch_info = propagate(prop_method, model_info, batch_info)
        batch_bound, batch_info = process_bound(prop_method, batch_bound, batch_out_spec, model_info, batch_info)
        all_bounds = backward_compute_bound(prop_method, [problem.input], model_info, nominal_outputs)
        plot_bounds(all_bounds, model_info, save_path; vis_center=true, save_bound=false)
        # batch_result = check_inclusion(prop_method, problem.Flux_model, batch_input, batch_bound, batch_out_spec)
    end
    # println("results:")
    # println(batch_result)
end

function center(bound::LazySet)
    return LazySets.center(bound)
end
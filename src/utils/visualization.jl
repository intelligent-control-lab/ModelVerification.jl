using Plots
using JLD2
using FileIO

function compute_all_bound(prop_method::ForwardProp, batch_input, batch_output, model_info, out_and_bounds)

    batch_info = init_propagation(prop_method, batch_input, nothing, model_info)
    
    batch_info = get_all_layer_output_size(model_info, batch_info, size(get_center(batch_input[1])))

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
    return all_bounds, batch_info
end

function get_all_layer_output_size(model_info, batch_info, input_size)
    # @show model_info
    # @show batch_info
    # println("Computing all layer output size")

    @assert length(model_info.start_nodes) == 1
    # @assert length(model_info.node_nexts[model_info.start_nodes[1]]) == 1
    # println(batch_info)
    batch_size = 1
    batch_info[model_info.start_nodes[1]][:size_after_layer] = (input_size..., batch_size)
    
    # @show model_info.start_nodes[1]
    # @show (input_size..., batch_size)
    #BFS
    queue = Queue{Any}()                            # Make an empty queue.
    # enqueue!(queue, model_info.node_nexts[model_info.start_nodes[1]][1])
    foreach(x -> enqueue!(queue, x), model_info.start_nodes)

    visit_cnt = Dict(node => 0 for node in model_info.all_nodes)    # Dictionary for the number of visits (cnt) for all nodes. 
    while !isempty(queue)                           # If queue is not empty!
        node = dequeue!(queue)                      # Take out a node from the queue. At first, it's one of the connecting nodes from the start nodes.
        
        # @show node
        for output_node in model_info.node_nexts[node]    # For each node connected from the current node.
            visit_cnt[output_node] += 1             # The output node is visited one more time!
            if length(model_info.node_prevs[output_node]) == visit_cnt[output_node]   # If all the previous nodes has been led to the `output_node`.
                enqueue!(queue, output_node)        # Add the `output_node` to the queue.
            end
        end
        
        node in model_info.start_nodes && continue # start nodes do not need computing bound

        if length(model_info.node_prevs[node]) == 2    # If this is  are two previous nodes connecting to the `node`
            if isa(model_info.node_layer[node], Union{typeof(+), typeof(-)})
                batch_info[node][:size_after_layer] = batch_info[model_info.node_prevs[node][1]][:size_after_layer]
                batch_info[node][:size_before_layer] = batch_info[model_info.node_prevs[node][1]][:size_after_layer]
            else
                error("Size propagation not implemented for: $model_info.node_layer[node]")
            end
        else
            prev_size = batch_info[model_info.node_prevs[node][1]][:size_after_layer]
            # @show model_info.node_prevs[node][1]
            # @show prev_size
            batch_info[node][:size_after_layer] = Flux.outputsize(model_info.node_layer[node], prev_size)
            batch_info[node][:size_before_layer] = prev_size
        end
        # @show node, batch_info[node][:size_after_layer]
    end
    return batch_info
end

function compute_all_bound(prop_method::BackwardProp, batch_input::AbstractVector,batch_output::AbstractVector, model_info, nominal_outputs)
    
    batch_size = length(batch_input)
    # @show keys(nominal_outputs)
    batch_info = init_propagation(prop_method, batch_input, get_linear_spec(batch_output), model_info)
    # @show reverse(model_info.all_nodes)
    # @show batch_info
    
    f_node = model_info.final_nodes[1]
    if !isnothing(batch_info[f_node][:bound].img_size)
        batch_info = get_all_layer_output_size(model_info, batch_info, batch_info[f_node][:bound].img_size)
    else
        batch_info = get_all_layer_output_size(model_info, batch_info, size(batch_info[f_node][:bound].batch_data_min)[1])
    end
    
    all_bounds = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)

    # TODO: need to conver this to BFS
    for node in model_info.all_nodes
        # @show "vis:", node
        if node in model_info.start_nodes
            continue
        end
        # @assert length(model_info.node_prevs[node]) == 1
        # prev_node = model_info.node_prevs[node][1]
        # @show node

        sub_model_info = get_sub_model(model_info, node)

        # @show node
        # @show sub_model_info.all_nodes

        # need to revise the following for image inputs with Convolution layers
        if isa(prop_method.pre_bound_method,BackwardProp)
            # @show model_info.node_layer[node]
            # @show keys(batch_info), keys(batch_info[node])
            # @show batch_info[node][:size_after_layer]
            
            # if isa(model_info.node_layer[node],Union{typeof(relu), MeanPool})
            #     n_out = batch_info[node][:size_after_layer]
            if length(batch_info[node][:size_after_layer]) == 4
                n_out = batch_info[node][:size_after_layer][1:3]
                # @show n_out
                n_out = n_out[1]*n_out[2]*n_out[3]
                
            else
                # dense weight, TODO: merge this else into if
                @assert length(batch_info[node][:size_after_layer]) == 2
                # @assert batch_info[node][:size_after_layer][1] == size(model_info.node_layer[node].weight)[1]
                n_out = batch_info[node][:size_after_layer][1]
            end
            I_spec = LinearSpec(repeat(Matrix(1.0I, n_out, n_out),1,1,batch_size), zeros(n_out, batch_size), false)
            # @show size(repeat(Matrix(1.0I, n_out, n_out),1,1,batch_size))
            # @show size(zeros(n_out, batch_size))
            if prop_method.use_gpu
                I_spec = LinearSpec(fmap(cu, I_spec.A), fmap(cu, I_spec.b), fmap(cu, I_spec.is_complement))
            end
            # @show size(I_spec.A)
            sub_out_spec, sub_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, I_spec, [all_bounds], sub_model_info, true)
            # println("keys: ", keys(sub_batch_info))
            # if haskey(sub_batch_info, "dense_0_relu")
            #     println("dense_0_relu low A:", sub_batch_info["dense_0_relu"])
            # end
            # println("dense_0_relu low A:", sub_batch_info["dense_0_relu"].lower_A_x)
            sub_batch_bound, sub_batch_info = propagate(prop_method.pre_bound_method, sub_model_info, sub_batch_info)
            sub_batch_bound, sub_batch_info = process_bound(prop_method.pre_bound_method, sub_batch_bound, sub_out_spec, sub_model_info, sub_batch_info)
            
            # @show sub_batch_bound
            l, u = compute_bound(sub_batch_bound) # reach_dim x batch 

        else
            # pre_bound_method is ForwardProp
            n_out = size(nominal_outputs[node][:out])[1]
            I_spec = LinearSpec(repeat(Matrix(1.0I, n_out, n_out),1,1,batch_size), zeros(n_out, batch_size), false)
            if prop_method.use_gpu
                I_spec = LinearSpec(fmap(cu, I_spec.A), fmap(cu, I_spec.b), fmap(cu, I_spec.is_complement))
            end
            sub_out_spec, sub_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, I_spec, [all_bounds], sub_model_info)
            sub_batch_bound, sub_batch_info = propagate(prop_method.pre_bound_method, sub_model_info, sub_batch_info)
            sub_batch_bound, sub_batch_info = process_bound(prop_method.pre_bound_method, sub_batch_bound, sub_out_spec, sub_model_info, sub_batch_info)
            l, u = compute_bound(sub_batch_bound) # reach_dim x batch 
        end
        
        
        # @show node
        # @show l
        # @show u
        all_bounds[node][:out] = nominal_outputs[node][:out]
        all_bounds[node][:l] = l
        all_bounds[node][:u] = u
        for next_node in model_info.node_nexts[node]
            if next_node in model_info.activation_nodes
                all_bounds[next_node][:pre_lower] = l
                all_bounds[next_node][:pre_upper] = u
            end
        end
        # @show keys(all_bounds[node])
        # @show all_bounds
    end
    return all_bounds, batch_info
end

function plot_bounds(all_bounds, model_info, batch_info, save_path; vis_center=true, save_bound=false, plot_mode=:cr)

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
            @show node
            
            l, u = all_bounds[node][:l], all_bounds[node][:u]
            # @show size(l),batch_info[node][:size_after_layer]
            l, u = l |> cpu, u |> cpu
            # @show node
            # @show batch_info
            # @show batch_info[node]
            # @show batch_info[node][:size_after_layer]
            l = reshape(l, (batch_info[node][:size_after_layer]))
            u = reshape(u, (batch_info[node][:size_after_layer]))

            println("saving visualized bound: ", save_path * string(i) * "_" * node * ".png")
            
            @show size(l)

            # out_center = ndims(l) == 4 ? (u + l)[:,:,1,1]./2 : reshape((u + l)./2, :,1)
            # @show size(out_center)
            # @show all_bounds[node]
            # @show size(all_bounds[node][:out])
            out_center = all_bounds[node][:out]
            out_radius = ndims(l) == 4 ? (u - l)[:,:,1,1] : reshape(u - l, :,1)
            
            out_l = ndims(l) == 4 ? l[:,:,1,1] : reshape(l, :,1)
            out_u = ndims(u) == 4 ? u[:,:,1,1] : reshape(u, :,1)

            out_center = ndims(out_center) == 4 ? out_center[:,:,1,1] : reshape(out_center, :,1)
            
            @show size(out_l)
            @show size(out_center)
            
            # @show size(out_center)
            # @show plot_mode
            if plot_mode == :curve
                p1 = plot(range(1,length(out_l)), [out_l out_center out_u], label=["lower" "center" "upper"])
                title!(string(i) * "_" * node)
                savefig(save_path * string(i) * "_" * node * ".png")
            elseif plot_mode == :lucr
                global_min = minimum(out_l)
                global_max = maximum(out_u)
                pl = heatmap(out_l, c = :ice, clims=(global_min, global_max))    
                title!(string(i) * "_" * node * "_lower")
                pu = heatmap(out_u, c = :ice, clims=(global_min, global_max))
                title!(string(i) * "_" * node * "_upper")
                p1 = heatmap(out_l - out_center, c = :ice)    
                title!(string(i) * "_" * node * "_lower - center")
                p2 = heatmap(out_u - out_center, c = :ice)
                title!(string(i) * "_" * node * "_upper - center")
                savefig(save_path * string(i) * "_" * node * ".png")
                p3 = heatmap(out_center)            
                title!(string(i) * "_" * node * "_center")
                p4 = heatmap(out_radius, c = :ice)
                title!(string(i) * "_" * node * "_bound_size")
                plot(pl, pu, p1, p2, p3, p4, layout=(3,2), size = (800,1000))
                savefig(save_path * string(i) * "_" * node * ".png")
                # @assert all(out_l .<= out_center)
                # @assert all(out_u .>= out_center)
            elseif plot_mode == :lu
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
                    plot_mode=:curve
                    )
    
    model_info, processed_problem = prepare_problem(search_method, split_method, prop_method, problem)

    # for node in model_info.all_nodes
    #     println(node, "->", model_info.node_nexts[node])
    # end

    processed_batch_input = [processed_problem.input]
    # processed_batch_outspec = [processed_problem.output]

    center_input = get_center(problem.input)
    input_size = size(center_input)
    # @show center_input
    original_batch_input = reshape(center_input, (input_size..., 1))
    nominal_outputs = compute_output(model_info, original_batch_input) # useful for infer output shapes
    
    all_bounds, batch_info = compute_all_bound(prop_method, processed_batch_input, [processed_problem.output], model_info, nominal_outputs)

    plot_bounds(all_bounds, model_info, batch_info, save_path; vis_center=vis_center, save_bound=save_bound, plot_mode=plot_mode)
    return all_bounds
end

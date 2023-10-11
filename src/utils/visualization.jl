using Plots
using JLD2
using FileIO
function visualize_propagate(prop_method::PropMethod, model_info, batch_info, save_path; vis_center=true, save_bound=false)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    #BFS
    dir = dirname(save_path)
    if !isdir(dir)
        mkdir(dir)
    end

    queue = Queue{Any}()
    enqueue_nodes!(prop_method, queue, model_info)
    out_cnt = Dict(node => 0 for node in model_info.all_nodes)
    visit_cnt = Dict(node => 0 for node in model_info.all_nodes)
    i = 0

    SNRs = []

    while !isempty(queue)
        i += 1
        node = dequeue!(queue)
        batch_info[:current_node] = node

        for output_node in father_nodes(prop_method, model_info, node)
            visit_cnt[output_node] += 1
            if all_prevs_in(prop_method, model_info, output_node, visit_cnt[output_node])
                enqueue!(queue, output_node)
            end
        end

        if has_two_reach_node(prop_method, model_info, node)
            batch_bound = propagate_skip_method(prop_method, model_info, batch_info, node)
            println(node)
            println(model_info.node_layer[node])
            batch_out = compute_out_skip(prop_method, model_info, batch_info, node)
        else
            stats = @timed batch_bound = propagate_layer_method(prop_method, model_info, batch_info, node)
            println("prop time:", stats.time)
            stats = @timed batch_out = compute_out_layer(prop_method, model_info, batch_info, node)
            println("comp time:", stats.time)
        end
        batch_info[node][:bound] = batch_bound
        batch_info[node][:out] = batch_out
        
        
        if !isnothing(save_path)
            println("saving visualized bound: ", save_path * string(i) * "_" * node * ".png")
            stats = @timed l, u = compute_bound(batch_bound[1])
            println("bound time:", stats.time) 

            batch_info[node][:l] = l
            batch_info[node][:u] = u
            
            img = ndims(batch_out) == 4 ? batch_out[:,:,1,1] : reshape(batch_out, :,1)
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
                plot(p_lu)
                savefig(save_path * string(i) * "_" * node * ".png", size = (400,300))
            end
        end

        for input_node in children_nodes(prop_method, model_info, node)
            out_cnt[input_node] += 1
            if all_nexts_in(prop_method, model_info, input_node, out_cnt[input_node])
                pop!(batch_info,input_node) # remove passed node to save memories
            end
        end
    end

    plot(SNRs, xlabel = "layer", ylabel="SNR", legend=false, yaxis=:log)
    savefig(save_path * "_SNRs_log.png")

    # plot(SNRs)
    # savefig(save_path * "_SNRs.png")

    return batch_info
end


function compute_out_skip(prop_method::ForwardProp, model_info, batch_info, node)
    input_node1 = model_info.node_prevs[node][1]
    input_node2 = model_info.node_prevs[node][2]
    batch_out1 = haskey(batch_info[input_node1], :out) ? batch_info[input_node1][:out] : LazySets.center(batch_info[input_node1][:bound][1])
    batch_out2 = haskey(batch_info[input_node2], :out) ? batch_info[input_node2][:out] : LazySets.center(batch_info[input_node2][:bound][1])
    return model_info.node_layer[node](batch_out1, batch_out2)
end

function compute_out_layer(prop_method::ForwardProp, model_info, batch_info, node)
    input_node1 = model_info.node_prevs[node][1]
    batch_out1 = haskey(batch_info[input_node1], :out) ? batch_info[input_node1][:out] : LazySets.center(batch_info[input_node1][:bound][1])
    return model_info.node_layer[node](batch_out1)
end

function visualize(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem, save_path; vis_center=true, save_bound=false)
    model_info, problem = prepare_problem(search_method, split_method, prop_method, problem)
    batch_out_spec, batch_info = prepare_method(prop_method, [problem.input], [problem.output], model_info)
    batch_info = visualize_propagate(prop_method, model_info, batch_info, save_path; vis_center=vis_center, save_bound=save_bound)
end
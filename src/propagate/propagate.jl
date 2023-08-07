enqueue_nodes!(prop_method::ForwardProp, queue, model_info) = enqueue!(queue, vcat([model_info.node_nexts[s] for s in model_info.start_nodes]...)...)
#enqueue_nodes!(prop_method::BackwardProp, queue, model_info) = enqueue!(queue, vcat([model_info.node_nexts[s] for s in model_info.final_nodes]...)...)
enqueue_nodes!(prop_method::BackwardProp, queue, model_info) = enqueue!(queue, [s for s in model_info.final_nodes]...)

all_prevs_in(prop_method::ForwardProp, model_info, output_node, cnt) = (cnt == length(model_info.node_prevs[output_node]))
all_prevs_in(prop_method::BackwardProp, model_info, output_node, cnt) = (cnt == length(model_info.node_nexts[output_node]))

has_two_reach_node(prop_method::ForwardProp, model_info, node) = (length(model_info.node_prevs[node]) == 2)
has_two_reach_node(prop_method::BackwardProp, model_info, node) = (length(model_info.node_nexts[node]) == 2)

father_nodes(prop_method::ForwardProp, model_info, node) = model_info.node_nexts[node]
father_nodes(prop_method::BackwardProp, model_info, node) = model_info.node_prevs[node]

children_nodes(prop_method::ForwardProp, model_info, node) = model_info.node_prevs[node]
children_nodes(prop_method::BackwardProp, model_info, node) = model_info.node_nexts[node]

all_nexts_in(prop_method::ForwardProp, model_info, output_node, cnt) = (cnt == length(model_info.node_nexts[output_node]))
all_nexts_in(prop_method::BackwardProp, model_info, output_node, cnt) = (cnt == length(model_info.node_prevs[output_node]))

function propagate(prop_method::PropMethod, model_info, batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    #BFS
    queue = Queue{Any}()
    enqueue_nodes!(prop_method, queue, model_info)
    visit_cnt = Dict(node => 0 for node in model_info.all_nodes)
    while !isempty(queue)
        node = dequeue!(queue)
        batch_info[:current_node] = node

        # if isa(prop_method, AlphaCrown)
        #     println(node)
        # end
       
        for output_node in father_nodes(prop_method, model_info, node)
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
        batch_info[node][:bound] = batch_bound
    end

    batch_bound = batch_info[model_info.final_nodes[1]][:bound]
    return batch_bound, batch_info
end


function propagate_skip_method(prop_method::ForwardProp, model_info, batch_info, node)
    input_node1 = model_info.node_prevs[node][1]
    input_node2 = model_info.node_prevs[node][2]
    batch_bound1 = batch_info[input_node1][:bound]
    batch_bound2 = batch_info[input_node2][:bound]
    batch_bound = propagate_skip_batch(prop_method, model_info.node_layer[node], batch_bound1, batch_bound2, batch_info)
    return batch_bound
end


function propagate_skip_method(prop_method::BackwardProp, model_info, batch_info, node)
    if !(node in model_info.start_nodes)
        output_node1 = model_info.node_nexts[node][1]
        output_node2 = model_info.node_nexts[node][2]
        batch_bound1 = batch_info[output_node1][:bound]
        batch_bound2 = batch_info[output_node2][:bound]
        batch_bound = propagate_skip_batch(prop_method, model_info.node_layer[node], batch_bound1, batch_bound2, batch_info)
    else
        return nothing
    end

    return batch_bound
end


function propagate_layer_method(prop_method::ForwardProp, model_info, batch_info, node)
    input_node = model_info.node_prevs[node][1]
    batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[input_node][:bound], batch_info)
    return batch_bound
end


function propagate_layer_method(prop_method::BackwardProp, model_info, batch_info, node)
    if !(node in model_info.start_nodes)
        if length(model_info.node_nexts[node]) != 0
            output_node = model_info.node_nexts[node][1]
            batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[output_node][:bound], batch_info)
        else #the node is final_node
            batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[node][:bound], batch_info)
        end
    else
        return nothing
    end

    return batch_bound
end


function propagate(prop_method::AdversarialAttack, model, batch_input, batch_out_spec, batch_info)
    # output: batch x ... x ...
    throw("unimplemented")
    # couterexample_result, batch_info = attack(prop_method, model, batch_input, batch_out_spec, batch_info)
    # return couterexample_result, batch_info
end

function propagate_linear_batch(prop_method::ForwardProp, layer, batch_reach::AbstractArray, batch_info)
    batch_reach_info = [propagate_linear(prop_method, layer, batch_reach[i], push!(batch_info, :batch_index => i)) for i in eachindex(batch_reach)]
    return batch_reach_info#map(first, batch_reach_info)
end

function propagate_act_batch(prop_method::ForwardProp, σ, batch_reach::AbstractArray, batch_info)
    batch_reach_info = [propagate_act(prop_method, σ, batch_reach[i], push!(batch_info, :batch_index => i)) for i in eachindex(batch_reach)]
    return batch_reach_info#map(first, batch_reach_info)
end

function propagate_skip_batch(prop_method::ForwardProp, layer, batch_reach1::AbstractArray, batch_reach2::AbstractArray, batch_info)
    batch_reach_info = [propagate_skip(prop_method, layer, batch_reach1[i], batch_reach2[i], push!(batch_info, :batch_index => i)) for i in eachindex(batch_reach1)]
    return batch_reach_info#map(first, batch_reach_info)
end

function is_activation(l)
    for f in NNlib.ACTIVATIONS
        isa(l, typeof(@eval NNlib.$(f))) && return true
    end
    return false
end

function propagate_layer_batch(prop_method, layer, batch_bound, batch_info)
    if is_activation(layer)
        batch_bound = propagate_act_batch(prop_method, layer, batch_bound, batch_info)
    else
        batch_bound = propagate_linear_batch(prop_method, layer, batch_bound, batch_info)
    end
    return batch_bound
end

function backward_layer(prop_method, layer, batch_bound)
    batch_bound = backward_linear(prop_method, layer, batch_bound)
    if hasfield(typeof(layer), :σ)
        batch_bound = backward_act(prop_method, layer.σ, batch_bound)
    end
    return batch_bound
end

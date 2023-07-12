function propagate(prop_method::PropMethod, model_info, batch_out_spec, batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    #BFS
    queue = Queue{Any}()
    enqueue!(queue, vcat([model_info.node_nexts[s] for s in model_info.start_nodes]...)...)
    visit_cnt = Dict(node => 0 for node in model_info.all_nodes)
    while !isempty(queue)
        node = dequeue!(queue)
        batch_info[:current_node] = node

        for output_node in model_info.node_nexts[node]
            visit_cnt[output_node] += 1
            if visit_cnt[output_node] == length(model_info.node_prevs[output_node])
                enqueue!(queue, output_node)
            end
        end

        if length(model_info.node_prevs[node]) == 2
            input_node1 = model_info.node_prevs[node][1]
            input_node2 = model_info.node_prevs[node][2]
            batch_bound1 = batch_info[input_node1][:bound]
            batch_bound2 = batch_info[input_node2][:bound]
            batch_bound = propagate_skip_batch(prop_method, model_info.node_layer[node], batch_bound1, batch_bound2, batch_info)
        else #length(batch_info[n][inputs] == 1
            input_node = model_info.node_prevs[node][1]
            batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[input_node][:bound], batch_info)
        end
        addbound(prop_method, node, batch_bound, batch_info)
    end

    batch_bound = batch_info[model_info.final_nodes[1]][:bound]
    return batch_bound, batch_info
end

function addbound(prop_method::ForwardProp, node, batch_bound, batch_info)
    push!(batch_info[node], :bound => batch_bound)
end

function addbound(prop_method::AlphaCrown, node, batch_bound, batch_info)
    if !isnothing(batch_bound.lower_A_x)
        if isnothing(batch_info[node][:bound].lower_A_x)
            # First A added to this node.
            new_node_lA = batch_bound.lower_A_x
        else
            new_node_lA = batch_info[node][:bound].lower_A_x .+ batch_bound.lower_A_x
        end
    else
        new_node_lA = nothing
    end

    if !isnothing(batch_bound.upper_A_x)
        if isnothing(batch_info[input_node][:uA])
            # First A added to this node.
            new_node_uA = batch_bound.upper_A_x
        else
            new_node_uA = batch_info[node][:bound].upper_A_x .+ batch_bound.upper_A_x
        end
    else
        new_node_uA = nothing  
    end
    batch_Low = batch_info[node][:bound].batch_Low
    batch_Up = batch_info[node][:bound].batch_Up
    lower_A_W = batch_info[node][:bound].lower_A_W
    upper_A_W = batch_info[node][:bound].upper_A_W
    lower_bias = batch_info[node][:bound].lower_bias
    upper_bias = batch_info[node][:bound].upper_bias
    new_bound = AlphaCrownBound(batch_Low, batch_Up, new_node_lA, new_node_uA, lower_A_W, upper_A_W, lower_bias, upper_bias)
    push!(batch_info[node], :bound => new_bound)
end

function get_degrees(prop_method::BackwardProp, node, batch_info)
    degrees = Dict()
    push!(batch_info[node], "bounded" => false)
    queue = Queue{Any}()
    enqueue!(queue, node)
    while !isempty(queue)
        node = dequeue!(queue)
        if !isnothing(model_info.node_prevs[node])
            for input_node in model_info.node_prevs[node]
                if haskey(degrees, input_node)
                    push!(degrees, input_node => degrees[input_node] + 1)
                else
                    push!(degrees, input_node => 1)
                end
                if batch_info[input_node]["bounded"]
                    push!(batch_info[input_node], "bounded" => false)
                    enqueue!(queue, input_node)
                end
            end
        end
    end
    return degrees
end
  
function propagate(prop_method::AdversarialAttack, model, batch_input, batch_out_spec, batch_info)
    # output: batch x ... x ...
    throw("unimplemented")
    # couterexample_result, batch_info = attack(prop_method, model, batch_input, batch_out_spec, batch_info)
    # return couterexample_result, batch_info
end


function forward(model, batch_input::AbstractArray)
    input_size = [] #input_size is a list
    for layer in (model.layers)
        layer_input_size = [size(batch_input, i) for i in 1:ndims(batch_input)-2] #only get the input size of the layer, the last 2 dims if channel and batchsize  
        push!(input_size, layer_input_size)
        batch_input = layer(batch_input)
    end
    return input_size
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




#= function add_bound(node, input_node, lA, uA, batch_info)
    if !isnothing(lA)
        if isnothing(batch_info[input_node][:lA])
            # First A added to this node.
            push!(batch_info[input_node], "lA" => lA)
        else
            #node_pre.zero_lA_mtx = node_pre.zero_lA_mtx and node.zero_backward_coeffs_l
            new_node_lA = batch_info[input_node][:lA] .+ lA
            push!(batch_info[input_node], "lA" => new_node_lA)
        end
    end
    if !isnothing(uA)
        if isnothing(batch_info[input_node][:uA])
            # First A added to this node.
            push!(batch_info[input_node], "uA" => uA)
        else
            #node_pre.zero_lA_mtx = node_pre.zero_lA_mtx and node.zero_backward_coeffs_l
            new_node_uA = batch_info[input_node][:uA] .+ uA
            push!(batch_info[input_node], "uA" => new_node_uA)
        end
    end
end=#


#= function concretize(lb, ub, batch_info, model_info)
    node = model_info["start_nodes"][1]
    if haskey(batch_info[node], "perturbation_info") #the node need to be perturbated
        if model_info["bound_lower"]
            lb = lb .+ ptb_concretize(model_info["model_inputs"], batch_info[node][:lA], -1, batch_info, model_info)
        else
            lb = nothing
        end
        if model_info["bound_upper"]
            ub = ub .+ ptb_concretize(model_inputs, batch_info[node][:uA], +1)
        else
            ub = nothing
        end    
    else #the node doesn't need to be perturbated
    end
end =#

#= function propagate(prop_method::BackwardProp, model, batch_bound, batch_out_spec, batch_info)
    # output: batch x ... x ...
    # dfs start from model.output_nodes
    
    input_size = forward(model, batch_input) #using forward to get the input size of the layers in the model
    layer_index = 0
    for layer in reverse(model.layers)
        if isa(layer, SkipConnection)
            batch_bound, batch_bias, batch_info = propagate(prop_method, layer.layers, batch_input, batch_bound, batch_info)
        elseif isa(layer, Conv)
            conv_input_size = input_size[end-layer_index] #get the original forward input size 
            batch_bound, batch_bias, batch_info = bound_backward(layer, conv_input_size, batch_bound, batch_info)
        end
        layer_index += 1
    end
    return batch_bound, batch_info
end =#

#= function IBP_general(node = nothing, model_info, C, delete_bounds_after_use = false)
    if !model_info.bound_opts.loss_fusion
        res = IBP_loss_fusion(node, C)
        if !isnothing(res)  
            return res
        end
    end

    if !node.ptb && hasproperty(!node, :forward_value)
        node.lower, node.upper = node.interval = (node.forward_value, node.forward_value)

    to_be_deleted_bounds = []
    if hasproperty(node, :interval)
        for n in node.inputs
            if !hasproperty(n, :interval)
                # Node n does not have interval bounds, so compute it.
                IBP_general(n, model_info, C, delete_bounds_after_use)
                push!(to_be_deleted_bounds, n)
        inp = [n_pre.interval for n_pre in node.inputs]
        if !isnothing(C) && node.type == "dense" and !node.inputs[2].ptb #maybe this "if" is useless
            # merge the last Dense node with the specification, available when weights of this layer are not perturbed
            ret = default_interval_propagate(node.layer, inp, C=C)
            _delete_unused_bounds(to_be_deleted_bounds)
            return ret
        else
            node.interval = interval_propagate(node.layer, inp)

        node.lower = node.interval[1]
        node.upper = node.interval[2]
        if isinstance(node.lower, torch.Size):
            node.lower = torch.tensor(node.lower)
            node.interval = (node.lower, node.upper)
        if isinstance(node.upper, torch.Size):
            node.upper = torch.tensor(node.upper)
            node.interval = (node.lower, node.upper)

    if C is not None:
        _delete_unused_bounds(to_be_deleted_bounds)
        return BoundLinear.interval_propagate(None, node.interval, C=C)
    else:
        _delete_unused_bounds(to_be_deleted_bounds)
        return node.interval
    end
end


function _delete_unused_bounds(node_list)
    """Delete bounds from input layers after use to save memory. Used when
    sparse_intermediate_bounds_with_ibp is true."""
    if delete_bounds_after_use
        for n in node_list
            n.interval = nothing # n must be a mutable struct
            n.lower = nothing
            n.upper =nothing
        end
    end
end

#= Args:
    inp: A list of the interval bound of input nodes.
    Generally, for each element `inp[i]`, `inp[i][0]` is the lower interval bound,
    and `inp[i][1]` is the upper interval bound.
Returns:
    bound: The interval bound of this node, in a same format as inp[i]. =#
#This interval_propagate is for non-layer node, like BoundConcat
function default_interval_propagate(layer, inp, C)
    """For unary monotonous functions or functions for altering shapes only but not values"""
    if length(inp) == 0
        return [forward(layer, inp, C), forward(layer, inp, C), nothing]
    elseif length(inp) == 1
        return [forward(layer, inp[1][1], C), self.forward(layer, inp[1][2], C), inp[1]]
    else
        error("default_interval_propagate only supports no more than 1 input node")
    end
end =#

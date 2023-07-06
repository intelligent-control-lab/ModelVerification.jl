#= function propagate(prop_method::ForwardProp, onnx_model_path, Flux_model, input_shape, batch_bound, batch_out_spec, batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    
    for layer in Flux_model.layers
        println(layer)
        if isa(layer, Parallel)
            par1_batch_bound, par1_batch_info = propagate(prop_method, onnx_model_path, layer.layers[:α], input_shape, batch_bound, batch_out_spec, batch_info)
            par2_batch_bound, par2_batch_info = propagate(prop_method, onnx_model_path, layer.layers[:β], input_shape, batch_bound, batch_out_spec, batch_info)
            batch_bound, batch_info = forward_skip_batch(prop_method, layer.connection, par1_batch_bound, par2_batch_bound, par1_batch_info, par2_batch_info)
        elseif isa(layer, SkipConnection)
            skip_batch_bound, skip_batch_info = propagate(prop_method, onnx_model_path, layer.layers, input_shape, batch_bound, batch_out_spec, batch_info)
            batch_bound, batch_info = forward_skip_batch(prop_method, layer.connection, batch_bound, skip_batch_bound, batch_info, skip_batch_info)
        else
            batch_bound, batch_info = forward_layer(prop_method, layer, batch_bound, batch_info)
            #println(layer, " reach ", low(batch_bound[1]), high(batch_bound[1]))
        end
        println(batch_bound)
    end

    return batch_bound, batch_info
end    =# 

function propagate(prop_method::ForwardProp, batch_info, global_info, batch_bound, batch_out_spec, aux_batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
   
    #BFS
    queue = Queue{Any}()
    start_nodes = global_info["start_nodes"]
    Isqueue = Dict()#determine whether the node is visited
    for node in start_nodes
        enqueue!(queue, node)
    end

    while !isempty(queue)
        node = dequeue!(queue)
        push!(Isqueue, node => false)
 
        if haskey(batch_info[node], "output_bound")
            continue
        end

        for output_node in batch_info[node]["outputs"]
            if haskey(batch_info[output_node], "output_bound") || (haskey(Isqueue, output_node) && Isqueue[output_node] == true)
                continue
            end
            enqueue!(queue, output_node)
            push!(Isqueue, output_node => true)
        end

        if isnothing(batch_info[node]["inputs"])
            current_batch_bound = batch_bound
            batch_bound, aux_batch_info = forward_layer(prop_method, batch_info[node]["layer"], current_batch_bound, aux_batch_info)
        elseif length(batch_info[node]["inputs"]) == 2
            input_node1 = batch_info[node]["inputs"][1]
            input_node2 = batch_info[node]["inputs"][2]
            if (!haskey(batch_info[input_node1], "output_bound") || !haskey(batch_info[input_node2], "output_bound")) && Isqueue[node] == false
                enqueue!(queue, node)
                push!(Isqueue, node => true)
                continue
            end
            current_batch_bound1 = batch_info[input_node1]["output_bound"]
            current_batch_bound2 = batch_info[input_node2]["output_bound"]
            aux_batch_info1 = batch_info[input_node1]["aux_batch_info"]
            aux_batch_info2 = batch_info[input_node2]["aux_batch_info"]
            batch_bound, aux_batch_info = forward_skip_batch(prop_method, batch_info[node]["layer"], current_batch_bound1, current_batch_bound2, aux_batch_info1, aux_batch_info2)
        else #length(batch_info[node][inputs] == 1
            input_node = batch_info[node]["inputs"][1]
            if !haskey(batch_info[input_node], "output_bound") && Isqueue[node] == false
                enqueue!(queue, node)
                push!(Isqueue, node => true)
                continue
            end
            current_batch_bound = batch_info[input_node]["output_bound"]
            aux_batch_info = batch_info[input_node]["aux_batch_info"]
            batch_bound, aux_batch_info = forward_layer(prop_method, batch_info[node]["layer"], current_batch_bound, aux_batch_info)
        end
        push!(batch_info[node], "output_bound" => batch_bound)
        push!(batch_info[node], "aux_batch_info" => aux_batch_info)
    end     

    final_nodes = global_info["final_nodes"][1]
    batch_bound = batch_info[final_nodes]["output_bound"]
    aux_batch_info = batch_info[final_nodes]["aux_batch_info"]
    

    return batch_bound, aux_batch_info 
end     

function propagate(prop_method::BackwardProp, C, node, unstable_idx, unstable_size, batch_info, global_info)
    for node in global_info["all_nodes"]
        push!(batch_info[node], "lA" => nothing)
        push!(batch_info[node], "uA" => nothing)
        push!(batch_info[node], "bounded" => true)
    end
    
    push!(batch_info[node], "lA" => global_info["bound_lower"] ? C : nothing)
    push!(batch_info[node], "uA" => global_info["bound_upper"] ? C : nothing)
    lb = ub = 0 #lb, ub => lower bound, upper bound
    degree_out = get_degrees(node, batch_info, global_info)
    C, batch_size, output_dim, output_shape = preprocess_C(C, node, batch_info, global_info)#size(C)=(10, 9, 1) 
    #batch_size = 1, output_dim = 9, output_shape = [-1] 
    queue = Queue{Any}()
    enqueue!(queue, node)

    while !isempty(queue)
        n = dequeue!(queue)
        push!(batch_info[n], "bounded" => true) #means this node has been computed bound

        for input_node in batch_info[n]["inputs"]
            degree_out[input_node] -= 1
            if degree_out[input_node] == 0
                enqueue!(queue, input_node)
            end
        end

        if !batch_info[n]["ptb"] #Haven't finish
            if !haskey(batch_info[n], "forward_value")
                get_forward_value(n)
            end
            lb, ub = add_constant_node(lb, ub, n, batch_info, global_info)
            continue
        end
        
        if isa(typeof(batch_info[n]["layer"]), typeof(relu))
            A, lower_b, upper_b = bound_backward(batch_info[n]["layer"], n, node, unstable_idx)
        elseif is_activation(batch_info[n]["layer"])   
        else
            A, lower_b, upper_b = bound_backward(batch_info[n]["layer"], node, batch_info, global_info) 
        end

        lb = lb .+ lower_b
        ub = ub .+ upper_b

        for (i, input_node) in enumerate(batch_info[node]["inputs"])
            add_bound(node, input_node, A[i][1], A[i][2], batch_info, global_info)
        end
    end

    lb, ub = concretize(lb, ub, batch_info, global_info) #add perturbation, haven't finish

end

function get_degrees(node, batch_info, global_info)
    degrees = Dict()
    push!(batch_info[node], "bounded" => false)
    queue = Queue{Any}()
    enqueue!(queue, node)
    while !isempty(queue)
        node = dequeue!(queue)
        if !isnothing(batch_info[node]["inputs"])
            for input_node in batch_info[node]["inputs"]
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

function preprocess_C(C, node, batch_info, global_info)
    batch_size = size(C)[end]
    output_dim = size(C)[2]
    output_shape = [-1]
    return C, batch_size, output_dim, output_shape
end

function add_bound(node, input_node, lA, uA, batch_info, global_info)
    if !isnothing(lA)
        if isnothing(batch_info[input_node]["lA"])
            # First A added to this node.
            push!(batch_info[input_node], "lA" => lA)
        else
            #node_pre.zero_lA_mtx = node_pre.zero_lA_mtx and node.zero_backward_coeffs_l
            new_node_lA = batch_info[input_node]["lA"] .+ lA
            push!(batch_info[input_node], "lA" => new_node_lA)
        end
    end
    if !isnothing(uA)
        if isnothing(batch_info[input_node]["uA"])
            # First A added to this node.
            push!(batch_info[input_node], "uA" => uA)
        else
            #node_pre.zero_lA_mtx = node_pre.zero_lA_mtx and node.zero_backward_coeffs_l
            new_node_uA = batch_info[input_node]["uA"] .+ uA
            push!(batch_info[input_node], "uA" => new_node_uA)
        end
    end
end


function concretize(lb, ub, batch_info, global_info)
    node = global_info["start_nodes"][1]
    if haskey(batch_info[node], "perturbation_info") #the node need to be perturbated
        if global_info["bound_lower"]
            lb = lb .+ ptb_concretize(global_info["model_inputs"], batch_info[node]["lA"], -1, batch_info, global_info)
        else
            lb = nothing
        end
        if global_info["bound_upper"]
            ub = ub .+ ptb_concretize(model_inputs, batch_info[node]["uA"], +1)
        else
            ub = nothing
        end    
    else #the node doesn't need to be perturbated
    end
end

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


#= function IBP_general(node = nothing, global_info, C, delete_bounds_after_use = false)
    if !global_info.bound_opts.loss_fusion
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
                IBP_general(n, global_info, C, delete_bounds_after_use)
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



function forward_linear_batch(prop_method::ForwardProp, layer, batch_reach::AbstractArray, batch_info::AbstractArray)
    batch_reach_info = [forward_linear(prop_method, layer, reach, info) for (reach, info) in zip(batch_reach, batch_info)]
    return map(first, batch_reach_info), map(last, batch_reach_info)
end

function forward_act_batch(prop_method::ForwardProp, σ, batch_reach::AbstractArray, batch_info::AbstractArray)
    batch_reach_info = [forward_act(prop_method, σ, reach, info) for (reach, info) in zip(batch_reach, batch_info)]
    return map(first, batch_reach_info), map(last, batch_reach_info)
end

function forward_skip_batch(prop_method::ForwardProp, layer, batch_reach1::AbstractArray, batch_reach2::AbstractArray, batch_info1::AbstractArray, batch_info2::AbstractArray)
    batch_reach_info = [forward_skip(prop_method, layer, batch_reach1[i], batch_reach2[i], batch_info1[i], batch_info2[i]) for i in eachindex(batch_reach1)]
    return map(first, batch_reach_info), map(last, batch_reach_info)
end

function is_activation(l)
    for f in NNlib.ACTIVATIONS
        isa(l, typeof(@eval NNlib.$(f))) && return true
    end
    return false
end

function forward_layer(prop_method, layer, batch_bound, batch_info)
    if is_activation(layer)
        batch_bound, batch_info = forward_act_batch(prop_method, layer, batch_bound, batch_info)
    else
        batch_bound, batch_info = forward_linear_batch(prop_method, layer, batch_bound, batch_info)
        #if hasfield(typeof(layer), :σ)
        #    batch_bound, batch_info = forward_act_batch(prop_method, layer.σ, batch_bound, batch_info)
        #end
    end
    return batch_bound, batch_info
end

function backward_layer(prop_method, layer, batch_bound, batch_info)
    batch_bound, batch_info = backward_linear(prop_method, layer, batch_bound, batch_info)
    if hasfield(typeof(layer), :σ)
        batch_bound, batch_info = backward_act(prop_method, layer.σ, batch_bound, batch_info)
    end
    return batch_bound, batch_info
end

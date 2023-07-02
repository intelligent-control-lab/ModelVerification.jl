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

function propagate(prop_method::ForwardProp, onnx_model_path, Flux_model, input_shape, batch_bound, batch_out_spec, aux_batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    @assert !isnothing(onnx_model_path) 

    #= if !isnothing(Flux_model) && !isnothing(input_shape)
        save(onnx_model_path, Flux_model, input_shape)
    end =#

    comp_graph = ONNXNaiveNASflux.load(onnx_model_path, infer_shapes=false)
    batch_info = Dict()
    global_info = Dict()
    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        if index == 1 # the vertex which index == 1 has no useful information, so it's output node will be the start node of the model
            push!(global_info, "start_node" => []) #creat a array for storing the start_nodes because maybe there are more than 1 start_node
            for node in outputs(vertex)
                push!(global_info["start_node"], NaiveNASflux.name(node))
            end
            continue
        end 
    
        new_dict = Dict() # store the information of this vertex 
        push!(new_dict, "vertex" => vertex)
        push!(new_dict, "layer" => NaiveNASflux.layer(vertex))
        push!(new_dict, "index" => index)
        push!(new_dict, "inputs" => inputs(vertex))# note: inputs(vertex)) is not a string, use NaiveNASflux.name convert them to string 
        push!(new_dict, "outputs" => outputs(vertex))# note: outputs(vertex)) is not a string, use NaiveNASflux.name convert them to string 
        
        #this part defines the layer of Flatten, relu, add and so on because the type of this layer's "layer(vertex)" is string(like "#213")
        if length(string(NaiveNASflux.name(vertex))) >= 7 && string(NaiveNASflux.name(vertex))[1:7] == "Flatten" 
            push!(new_dict, "layer" => Flux.flatten)
        elseif length(string(NaiveNASflux.name(vertex))) >= 4 && string(NaiveNASflux.name(vertex))[1:4] == "relu" 
            push!(new_dict, "layer" => NNlib.relu)
        elseif length(string(NaiveNASflux.name(vertex))) >= 3 && string(NaiveNASflux.name(vertex))[1:3] == "add" 
            push!(new_dict, "layer" => +)
        end

        push!(batch_info, NaiveNASflux.name(vertex) => new_dict) #new_dict belongs to batch_info
        if length(outputs(vertex)) == 0  #the final node has no output nodes
            global_info["final_node"] = NaiveNASflux.name(vertex)
        end
    end

    #BFS
    queue = Queue{Any}()
    start_node = global_info["start_node"]
    Isvisit = Dict()#determine whether the node is visited
    for node in start_node
        enqueue!(queue, node)
        batch_info[node]["inputs"] = nothing #start_nodes have no input nodes
    end

    while !isempty(queue)
        node = dequeue!(queue)
        if haskey(Isvisit, node) #means that this node has been visited
            continue
        end
        push!(Isvisit, node => true)
        for node in batch_info[node]["outputs"]
            if haskey(Isvisit, node) #means that this node has been visited
                continue
            end
            enqueue!(queue, NaiveNASflux.name(node))
        end

        if isnothing(batch_info[node]["inputs"])
            current_batch_bound = batch_bound
            batch_bound, aux_batch_info = forward_layer(prop_method, batch_info[node]["layer"], current_batch_bound, aux_batch_info)
        elseif length(batch_info[node]["inputs"]) == 2
            input_node1 = NaiveNASflux.name(batch_info[node]["inputs"][1])
            input_node2 = NaiveNASflux.name(batch_info[node]["inputs"][2])
            current_batch_bound1 = batch_info[input_node1]["output_bound"]
            current_batch_bound2 = batch_info[input_node2]["output_bound"]
            aux_batch_info1 = batch_info[input_node1]["aux_batch_info"]
            aux_batch_info2 = batch_info[input_node2]["aux_batch_info"]
            batch_bound, aux_batch_info = forward_skip_batch(prop_method, batch_info[node]["layer"], current_batch_bound1, current_batch_bound2, aux_batch_info1, aux_batch_info2)
        else #length(batch_info[node][inputs] == 1
            input_node = NaiveNASflux.name(batch_info[node]["inputs"][1])
            current_batch_bound = batch_info[input_node]["output_bound"]
            aux_batch_info = batch_info[input_node]["aux_batch_info"]
            batch_bound, aux_batch_info = forward_layer(prop_method, batch_info[node]["layer"], current_batch_bound, aux_batch_info)
        end
        push!(batch_info[node], "output_bound" => batch_bound)
        push!(batch_info[node], "aux_batch_info" => aux_batch_info)
    end     

    final_node = global_info["final_node"]
    batch_bound = batch_info[final_node]["output_bound"]
    aux_batch_info = batch_info[final_node]["aux_batch_info"]

    return batch_bound, aux_batch_info 
end     


function propagate(prop_method::BackwardProp, model, batch_bound, batch_out_spec, batch_info)
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
        if hasfield(typeof(layer), :σ)
            batch_bound, batch_info = forward_act_batch(prop_method, layer.σ, batch_bound, batch_info)
        end
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

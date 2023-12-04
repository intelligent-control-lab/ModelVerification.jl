"""
    enqueue_nodes!(prop_method::ForwardProp, queue, model_info)

Inserts the nodes connected from the starting node into the given `queue` for 
`ForwardProp` methods.
"""
enqueue_nodes!(prop_method::ForwardProp, queue, model_info) = enqueue!(queue, vcat([model_info.node_nexts[s] for s in model_info.start_nodes]...)...)

"""
    enqueue_nodes!(prop_method::BackwardProp, queue, model_info)

Inserts the final nodes into the given `queue` for `BackwardProp` methods.
"""
enqueue_nodes!(prop_method::BackwardProp, queue, model_info) = enqueue!(queue, [s for s in model_info.final_nodes]...)

"""
    output_node(prop_method::ForwardProp, model_info)    

Returns the final nodes of the model.
"""
output_node(prop_method::ForwardProp, model_info) = model_info.final_nodes[1]

"""
    output_node(prop_method::BackwardProp, model_info)

Returns the starting nodes of the model for `BackwardProp` methods. Since this 
is for `BackwardProp` methods, the starting nodes of the model are the output 
nodes.
"""
output_node(prop_method::BackwardProp, model_info) = model_info.node_nexts[model_info.start_nodes[1]][1]

"""
    next_nodes(prop_method::ForwardProp,  model_info, node)

Returns the next nodes of the `node` for `ForwardProp` methods.
"""
next_nodes(prop_method::ForwardProp,  model_info, node) = model_info.node_nexts[node]

"""
    next_nodes(prop_method::BackwardProp, model_info, node)

Returns the previous nodes of the `node` for `BackwardProp` methods. Since this 
is for `BackwardProp` methods, the previous nodes are the "next" nodes.
"""
next_nodes(prop_method::BackwardProp, model_info, node) = model_info.node_prevs[node]

"""
    prev_nodes(prop_method::ForwardProp,  model_info, node)

Returns the previous nodes of the `node` for `ForwardProp` methods.
"""
prev_nodes(prop_method::ForwardProp,  model_info, node) = model_info.node_prevs[node]

"""
    prev_nodes(prop_method::BackwardProp, model_info, node)

Returns the next nodes of the `node` for `BackwardProp` methods. Since this is 
for `BackwardProp` methods, the next nodes are the "previous" nodes.
"""
prev_nodes(prop_method::BackwardProp, model_info, node) = model_info.node_nexts[node]

"""
    all_nexts_in(prop_method, model_info, output_node, cnt)

Returns true if all of the next nodes of the `output_node` have been visited.
"""
all_nexts_in(prop_method, model_info, output_node, cnt) = (cnt == length(next_nodes(prop_method, model_info, output_node)))

"""
    all_prevs_in(prop_method, model_info, output_node, cnt)

Returns true if the `output_node` has been visited from all the previous nodes.
This function checks if all possible connections to the `output_node` has been 
made in the propagation procedure.
For example, given a node X, say that there are 5 different nodes that are 
mapped to X. Then, if the node X has been visited 5 times, i.e., `cnt` == 5, 
it means that all the previous nodes of X has been outputted to X.
"""
all_prevs_in(prop_method, model_info, output_node, cnt) = (cnt == length(prev_nodes(prop_method, model_info, output_node)))

"""
    has_two_reach_node(prop_method, model_info, node)

Checks whether there are two nodes connected to the current `node`, i.e., there 
are two previous nodes. This function is used to check if there are skip 
connections.
"""
has_two_reach_node(prop_method, model_info, node) = (length(prev_nodes(prop_method, model_info, node)) == 2)

"""
    propagate(prop_method::PropMethod, model_info, batch_info)

Propagates through the model using the specified `prop_method`. 
The propagation algorithm is as follows:

1. Add the connecting nodes of the start nodes, i.e., nodes after the start 
nodes, into a queue.
2. While the queue is not empty:
    a. Pop a node from the queue.
    b. For each node connected from the current node, i.e., for each output 
        node:
        i. Increment the visit count to the output node.
        ii. If the visit count equals the number of nodes connected from the 
            output node, i.e., visit count == previous nodes of the output node, 
            add the output node to the queue.
    c. Propagate through the current node accordingly.
    d. Add information about the bound of the node to `batch_info`.
3. Return the bound of the output node(s).

In step 2(a)(ii), the function adds the output node to the queue since all the 
previous nodes of the output node have been processed. Thus, the output node is 
now the node of interest. In step 2(c), the propagation works based on the 
propagation method (`prop_method`), which depends on the geometric 
representation of the safety specifications and the activation function of each 
layer.

## Arguments
- `prop_method` (`PropMethod`): Propagation method used for the verification 
    process. This is one of the solvers used to verify the given model.
- `model_info`: Structure containing the information of the neural network to be 
    verified.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `batch_bound`: Bound of the output node, i.e., the final bound.
- `batch_info`: Same as the input `batch_info`, with additional information on 
    the bound of each node in the model.
"""
function propagate(prop_method::PropMethod, model_info, batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    #BFS
    queue = Queue{Any}()                            # Make an empty queue.
    enqueue_nodes!(prop_method, queue, model_info)  # Insert connecting nodes from the start nodes (there's only one start node).
    visit_cnt = Dict(node => 0 for node in model_info.all_nodes)    # Dictionary for the number of visits (cnt) for all nodes. 
    batch_bound = nothing                           # ?
    while !isempty(queue)                           # If queue is not empty!
        node = dequeue!(queue)                      # Take out a node from the queue. At first, it's one of the connecting nodes from the start nodes.
        batch_info[:current_node] = node            # Add a new key-value pair: `:current_node` => `node`
       
        for output_node in next_nodes(prop_method, model_info, node)    # For each node connected from the current node.
            visit_cnt[output_node] += 1             # The output node is visited one more time!
            if all_prevs_in(prop_method, model_info, output_node, visit_cnt[output_node])   # If all the previous nodes has been led to the `output_node`.
                enqueue!(queue, output_node)        # Add the `output_node` to the queue.
                                                    # We're essentially moving the propagation to the `output_node` since all the previous connecting nodes
                                                    # have been "processed"/propagated through. 
            end
        end

        if has_two_reach_node(prop_method, model_info, node)    # If there are two previous nodes connecting to the `node`.
            batch_bound = propagate_skip_method(prop_method, model_info, batch_info, node)
        else
            batch_bound = propagate_layer_method(prop_method, model_info, batch_info, node)
        end
        batch_info[node][:bound] = batch_bound      # Add information about the bound for the node.
    end
    batch_bound = batch_info[output_node(prop_method, model_info)][:bound]  # Bound of the output node! Final bound!
    return batch_bound, batch_info
end

"""
    propagate_skip_method(prop_method::ForwardProp, model_info, 
                          batch_info, node)

This function propagates the two sets of bounds of the preceding nodes from the 
provided `node` using the specified forward propagation method and layer 
operation. It invokes `propagate_skip_batch`, which subsequently calls  
`propagate_skip`. The function identifies the two previous nodes from the given
`node` in the computational graph, `model_info`, their bounds, and the layer 
operation of the node. Then, `propagate_skip` is invoked.

## Arguments
- `prop_method` (`ForwardProp`): Forward propagation method used for the 
    verification process. This is one of the solvers used to verify the given 
    model.
- `model_info`: Structure containing the information of the neural network to be 
    verified.
- `batch_info`: Dictionary containing information of each node in the model.
- `node`: The current node to be propagated through.

## Returns
- `batch_bound`: List of reachable bounds after propagating the two sets of
    bounds in `batch_reach` through the given `node`, following the propagation 
    method and the layer operation.
"""
function propagate_skip_method(prop_method::ForwardProp, model_info, batch_info, node)
    input_node1 = model_info.node_prevs[node][1]
    input_node2 = model_info.node_prevs[node][2]
    batch_bound1 = batch_info[input_node1][:bound]
    batch_bound2 = batch_info[input_node2][:bound]
    batch_bound = propagate_skip_batch(prop_method, model_info.node_layer[node], batch_bound1, batch_bound2, batch_info)
    return batch_bound
end

"""
    propagate_skip_method(prop_method::BackwardProp, model_info, 
                          batch_info, node)

This function propagates the two sets of bounds of the next nodes from the 
provided `node` using the specified backward propagation method and layer 
operation. It invokes `propagate_skip_batch`, which subsequently calls  
`propagate_skip`. The function identifies the two next nodes from the given
`node` in the computational graph, `model_info`, their bounds, and the layer 
operation of the node. Then, `propagate_skip` is invoked.

## Arguments
- `prop_method` (`BackwardProp`): Backward propagation method used for the 
    verification process. This is one of the solvers used to verify the given 
    model.
- `model_info`: Structure containing the information of the neural network to be 
    verified.
- `batch_info`: Dictionary containing information of each node in the model.
- `node`: The current node to be propagated through.

## Returns
- `batch_bound`: List of reachable bounds after propagating the two sets of
    bounds in `batch_reach` through the given `node`, following the propagation 
    method and the layer operation.                          
"""
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

"""
    propagate_layer_method(prop_method::ForwardProp, model_info, batch_info, node)

This function propagates the bounds of the preceding node from the provided node 
using the specified forward propagation method and layer operation. It invokes 
`propagate_layer_batch`, which subsequently calls either 
`propagate_linear_batch` or `propagate_act_batch`. The function identifies the 
previous node from the given node in the computational graph, `model_info`, its 
bound, and the layer operation of the node. Then, `propagate_layer_batch` 
ascertains if the layer operation is linear or includes activation functions 
like ReLU. Depending on this, `propagate_linear_batch` or `propagate_act_batch` 
is invoked.

## Arguments
- `prop_method` (`ForwardProp`): The forward propagation method employed for 
    verification. It is one of the solvers used to validate the specified model.
- `model_info`: Structure containing the information of the neural network to be 
    verified.
- `batch_info`: Dictionary containing information of each node in the model.
- `node`: The current node to be propagated through.

## Returns
- `batch_bound`: List of reachable bounds after propagating the set of input 
    bounds of the given `node`, following the propagation method and the linear 
    layer operation.
"""
function propagate_layer_method(prop_method::ForwardProp, model_info, batch_info, node)
    input_node = model_info.node_prevs[node][1]
    to = get_timer("Shared")
    @timeit to string(nameof(typeof(model_info.node_layer[node]))) batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[input_node][:bound], batch_info)
    return batch_bound
end

"""
    propagate_layer_method(prop_method::BackwardProp, model_info, batch_info, node)

This function propagates the bounds of the next node from the provided node 
using the specified forward propagation method and layer operation. It invokes 
`propagate_layer_batch`, which subsequently calls either 
`propagate_linear_batch` or `propagate_act_batch`. The function identifies the 
next node from the given node in the computational graph, `model_info`, its 
bound, and the layer operation of the node. Then, `propagate_layer_batch` 
ascertains if the layer operation is linear or includes activation functions 
like ReLU. Depending on this, `propagate_linear_batch` or `propagate_act_batch` 
is invoked.

## Arguments
- `prop_method` (`BackwardProp`): Backward propagation method used for the 
    verification process. This is one of the solvers used to verify the given 
    model.
- `model_info`: Structure containing the information of the neural network to be 
    verified.
- `batch_info`: Dictionary containing information of each node in the model.
- `node`: The current node to be propagated through.

## Returns
- `batch_bound`: List of reachable bounds after propagating the set of bounds in 
    `batch_reach` through the given `node`, following the propagation method and 
    the linear layer operation.
- `nothing` if the given `node` is a starting node of the model.
"""
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

"""
    propagate_linear_batch(prop_method::ForwardProp, layer, 
                           batch_reach::AbstractArray, batch_info)

Propagates each of the bound in the `batch_reach` array with the given forward 
propagation method, `prop_method`, through a linear layer. 

## Arguments
- `prop_method` (`ForwardProp`): Forward propagation method used for the 
    verification process. This is one of the solvers used to verify the given 
    model.
- `layer`: Identifies what type of operation is done at the layer. Here, its a 
    linear operation.
- `batch_reach` (`AbstractArray`): List of input specifications, i.e., bounds,  
    to be propagated through the given `layer`.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `batch_reach_info`: List of reachable bounds after propagating the set of 
    bounds in `batch_reach` through the given `layer`, following the propagation 
    method and the linear layer operation.
"""
function propagate_linear_batch(prop_method::ForwardProp, layer, batch_reach::AbstractArray, batch_info)
    batch_reach_info = [propagate_linear(prop_method, layer, batch_reach[i], push!(batch_info, :batch_index => i)) for i in eachindex(batch_reach)]
    return batch_reach_info#map(first, batch_reach_info)
end

"""
    propagate_act_batch(prop_method::ForwardProp, σ, 
                        batch_reach::AbstractArray, batch_info)

Propagates each of the bound in the `batch_reach` array with the given forward 
propagation method, `prop_method`, through an activation layer. 

## Arguments
- `prop_method` (`ForwardProp`): Forward propagation method used for the 
    verification process. This is one of the solvers used to verify the given 
    model.
- `σ`: Type of activation function, such as ReLU.
- `batch_reach` (`AbstractArray`): List of input specifications, i.e., bounds,  
    to be propagated through the given `layer`.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `batch_reach_info`: List of reachable bounds after propagating the set of 
    bounds in `batch_reach` through the given `layer`, following the propagation 
    method and the activation layer operation.                        
"""
function propagate_act_batch(prop_method::ForwardProp, σ, batch_reach::AbstractArray, batch_info)
    batch_reach_info = [propagate_act(prop_method, σ, batch_reach[i], push!(batch_info, :batch_index => i)) for i in eachindex(batch_reach)]
    return batch_reach_info#map(first, batch_reach_info)
end

"""
    propagate_skip_batch(prop_method::ForwardProp, layer, 
                         batch_reach1::AbstractArray, 
                         batch_reach2::AbstractArray, 
                         batch_info)

Propagates each combination of the bounds from the `batch_reach1` and 
`batch_reach2` arrays with the given forward propagation method, `prop_method`, 
through a skip connection. 

## Arguments
- `prop_method` (`ForwardProp`): Forward propagation method used for the 
    verification process. This is one of the solvers used to verify the given 
    model.
- `layer`: Identifies what type of operation is done at the layer. Here's a 
    bivariate operation is mainly used.
- `batch_reach1` (`AbstractArray`): First list of input specifications, i.e., 
    bounds, to be propagated through the given `layer`. This is the list of 
    bounds given by the first of the two previous nodes.
- `batch_reach2` (`AbstractArray`): Second list of input specifications, i.e., 
    bounds, to be propagated through the given `layer`. This is the list of 
    bounds given by the second of the two previous nodes.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `batch_reach_info`: List of reachable bounds after propagating the 
    bounds in `batch_reach1` and `batch_reach2` through the given `layer`, 
    following the propagation method and the layer operation.                         
"""
function propagate_skip_batch(prop_method::ForwardProp, layer, batch_reach1::AbstractArray, batch_reach2::AbstractArray, batch_info)
    batch_reach_info = [propagate_skip(prop_method, layer, batch_reach1[i], batch_reach2[i], push!(batch_info, :batch_index => i)) for i in eachindex(batch_reach1)]
    return batch_reach_info#map(first, batch_reach_info)
end

"""
    is_activation(l)

Returns true if the given layer `l` is an activation layer.

## Arguments
- `l`: Layer.

## Returns
- True if `l` is activation layer.
- False otherwise.
"""
function is_activation(l)
    for f in NNlib.ACTIVATIONS
        isa(l, typeof(@eval NNlib.$(f))) && return true
    end
    return false
end

"""
    propagate_layer_batch(prop_method, layer, batch_bound, batch_info)

Propagates through one layer. The given `layer` identifies what type of 
operation is performed. Operations such as Dense, BatchNorm, Convolution, and 
ReLU are supported. The `prop_method` denotes the solver (and thus the geometric 
representation for safety specification). The output of the propagation is the 
bounds of the given batch. 

## Arguments
- `prop_method`: Propagation method used for the verification process. This is 
    one of the solvers used to verify the given model.
- `layer`: Identifies what type of operation is done at the layer, such as 
    Dense, BatchNorm, Convolution, or ReLU.
- `batch_bound`: Bound of the input node (the previous node).
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `batch_bound`: The output bound after applying the operation of the node.
"""
function propagate_layer_batch(prop_method, layer, batch_bound, batch_info)
    to = get_timer("Shared")
    if is_activation(layer)
        @timeit to "propagate_act_batch" batch_bound = propagate_act_batch(prop_method, layer, batch_bound, batch_info)
    else
        @timeit to "propagate_linear_batch" batch_bound = propagate_linear_batch(prop_method, layer, batch_bound, batch_info)
    end
    return batch_bound
end

"""
    backward_layer(prop_method, layer, batch_bound)

(DEPRECATED)
"""
function backward_layer(prop_method, layer, batch_bound)
    batch_bound = backward_linear(prop_method, layer, batch_bound)
    if hasfield(typeof(layer), :σ)
        batch_bound = backward_act(prop_method, layer.σ, batch_bound)
    end
    return batch_bound
end

"""
    forward_linear_batch(prop_method::ForwardProp, layer, batch_reach::AbstractArray, batch_info::AbstractArray)

(DEPRECATED)
"""
function forward_linear_batch(prop_method::ForwardProp, layer, batch_reach::AbstractArray, batch_info::AbstractArray)
    batch_reach_info = [forward_linear(prop_method, layer, reach, info) for (reach, info) in zip(batch_reach, batch_info)]
    return map(first, batch_reach_info), map(last, batch_reach_info)
end

"""
    forward_act_batch(prop_method::ForwardProp, σ, batch_reach::AbstractArray, batch_info::AbstractArray)

    (DEPRECATED)
"""
function forward_act_batch(prop_method::ForwardProp, σ, batch_reach::AbstractArray, batch_info::AbstractArray)
    batch_reach_info = [forward_act(prop_method, σ, reach, info) for (reach, info) in zip(batch_reach, batch_info)]
    return map(first, batch_reach_info), map(last, batch_reach_info)
end

"""
    forward_skip_batch(prop_method::ForwardProp, layer, batch_reach1::AbstractArray, batch_reach2::AbstractArray, batch_info1::AbstractArray, batch_info2::AbstractArray)

(DEPRECATED)
"""
function forward_skip_batch(prop_method::ForwardProp, layer, batch_reach1::AbstractArray, batch_reach2::AbstractArray, batch_info1::AbstractArray, batch_info2::AbstractArray)
    batch_reach_info = [forward_skip(prop_method, layer, batch_reach1[i], batch_reach2[i], batch_info1[i], batch_info2[i]) for i in eachindex(batch_reach1)]
    return map(first, batch_reach_info), map(last, batch_reach_info)
end

"""
    forward_layer(prop_method, layer, batch_bound, batch_info)

(DEPRECATED)
"""
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

"""
    backward_layer(prop_method, layer, batch_bound, batch_info)

(DEPRECATED)
"""
function backward_layer(prop_method, layer, batch_bound, batch_info)
    batch_bound, batch_info = backward_linear(prop_method, layer, batch_bound, batch_info)
    if hasfield(typeof(layer), :σ)
        batch_bound, batch_info = backward_act(prop_method, layer.σ, batch_bound, batch_info)
    end
    return batch_bound, batch_info
end

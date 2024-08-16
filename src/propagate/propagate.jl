"""
    propagate(prop_method::PropMethod, model_info, batch_info)

Propagates through the model using the specified `prop_method`. 
The propagation algorithm is as follows:

1. Add the connecting nodes of the start nodes, i.e., nodes after the start 
nodes, into a queue.
2. While the queue is not empty:
    1. Pop a node from the queue.
    2. For each node connected from the current node, i.e., for each output 
        node:
        1. Increment the visit count to the output node.
        2. If the visit count equals the number of nodes connected from the 
            output node, i.e., visit count == previous nodes of the output node, 
            add the output node to the queue.
    3. Propagate through the current node accordingly.
    4. Add information about the bound of the node to `batch_info`.
3. Return the bound of the output node(s).

In step 2(1)(2), the function adds the output node to the queue since all the 
previous nodes of the output node have been processed. Thus, the output node is 
now the node of interest. In step 2(3), the propagation works based on the 
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
    visit_cnt = Dict(node => 0 for node in model_info.all_nodes)    # Dictionary for the number of visits (cnt) for all nodes. 
    
    # Insert connecting nodes from the start nodes (there's only one start node).
    # foreach(node -> enqueue_connected!(prop_method, model_info, queue, visit_cnt, node), start_nodes(prop_method, model_info))
    foreach(node -> enqueue!(queue, node), start_nodes(prop_method, model_info))
    
    to = get_timer("Shared")

    batch_bound = nothing                           # ?
    # println("propagating")
    while !isempty(queue)                           # If queue is not empty!
        node = dequeue!(queue)                      # Take out a node from the queue. At first, it's one of the connecting nodes from the start nodes.
        batch_info[:current_node] = node            # Add a new key-value pair: `:current_node` => `node`
        # println("prop: ", node, "    prev: ", prev_nodes(prop_method, model_info, node))
        # @show node
        enqueue_connected!(prop_method, model_info, queue, visit_cnt, node)
        @timeit to string(typeof(model_info.node_layer[node])) batch_bound = propagate_layer_method(prop_method, model_info, batch_info, node)
        batch_info[node][:bound] = batch_bound      # Add information about the bound for the node.
        
        # @show "add_0"
        # if haskey(batch_info, "add_0")
        #     for i in eachindex(batch_info["add_0"][:bound].lower_A_x)
        #         @show i, typeof(batch_info["add_0"][:bound].lower_A_x[i])
        #     end
        #     @assert length(batch_info["add_0"][:bound].lower_A_x) <2 || !isa(batch_info["add_0"][:bound].lower_A_x[2], BetaLayer)
        # end
        # println("---")
        # @show node
        # @show batch_bound
        @show node
    end
    # batch_bound = batch_info[output_node(prop_method, model_info)][:bound]  # Bound of the output node! Final bound!
    # @show output_node(prop_method, model_info)
    # @show batch_bound
    return batch_bound, batch_info
end

"""
    enqueue_connected!(prop_method::PropMethod, model_info, queue, visit_cnt, node)

Put all the connected node into the queue for propagation.
"""
function enqueue_connected!(prop_method::PropMethod, model_info, queue, visit_cnt, node)
    for output_node in next_nodes(prop_method, model_info, node)    # For each node connected from the current node.
        visit_cnt[output_node] += 1             # The output node is visited one more time!
        if all_prevs_in(prop_method, model_info, output_node, visit_cnt[output_node])   # If all the previous nodes has been led to the `output_node`.
            enqueue!(queue, output_node)        # Add the `output_node` to the queue.
                                                # We're essentially moving the propagation to the `output_node` since all the previous connecting nodes
                                                # have been "processed"/propagated through. 
        end
    end
end


"""
    start_nodes(prop_method::ForwardProp, queue, model_info)

Inserts the nodes connected from the starting node into the given `queue` for 
`ForwardProp` methods.
"""
start_nodes(prop_method::ForwardProp, model_info) = model_info.start_nodes

"""
    start_nodes(prop_method::BackwardProp, queue, model_info)

Inserts the final nodes into the given `queue` for `BackwardProp` methods.
"""
start_nodes(prop_method::BackwardProp, model_info) = model_info.final_nodes

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
output_node(prop_method::BackwardProp, model_info) = model_info.start_nodes[1]

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
    num_prevs(prop_method, model_info, node)

Checks whether there are two nodes connected to the current `node`, i.e., there 
are two previous nodes. This function is used to check if there are skip 
connections.
"""
num_prevs(prop_method, model_info, node) = length(prev_nodes(prop_method, model_info, node))


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
    @assert length(model_info.node_prevs[node]) == 2
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
    # There might be more than two next node for $node if the skip layer contains another skip layer.
    # output_node1 = model_info.node_nexts[node][1]
    # output_node2 = model_info.node_nexts[node][2]
    # batch_bound1 = batch_info[output_node1][:bound]
    # batch_bound2 = batch_info[output_node2][:bound]
    bounds = [batch_info[output_node][:bound] for output_node in model_info.node_nexts[node]]
    batch_bound = propagate_skip_batch(prop_method, Parallel, bounds, batch_info)
    batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_bound, batch_info)
    # if !(node in model_info.start_nodes)
        
    # else
    #     return batch_bound
    # end

    return batch_bound
end

"""
    propagate_layer_method(prop_method::ForwardProp, model_info, batch_info, node)

This function propagates the bounds of the preceding node from the provided node 
using the specified forward propagation method and layer operation. It invokes 
`propagate_layer_batch`, which subsequently calls either 
`propagate_layer_batch` or `propagate_layer_batch`. The function identifies the 
previous node from the given node in the computational graph, `model_info`, its 
bound, and the layer operation of the node. Then, `propagate_layer_batch` 
ascertains if the layer operation is linear or includes activation functions 
like ReLU. Depending on this, `propagate_layer_batch` or `propagate_layer_batch` 
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
    if length(model_info.node_prevs[node]) >= 2
        input_nodes = model_info.node_prevs[node]
        batch_bounds = [batch_info[node][:bound] for node in input_nodes]
        # [reach x batch] x parallel_dim
        batch_bound = propagate_parallel_batch(prop_method, model_info.node_layer[node], batch_bounds, batch_info)
    elseif length(model_info.node_prevs[node]) == 2
        input_node1 = model_info.node_prevs[node][1]
        input_node2 = model_info.node_prevs[node][2]
        batch_bound1 = batch_info[input_node1][:bound]
        batch_bound2 = batch_info[input_node2][:bound]
        batch_bound = propagate_skip_batch(prop_method, model_info.node_layer[node], batch_bound1, batch_bound2, batch_info)
    elseif length(model_info.node_prevs[node]) == 1
        input_node = model_info.node_prevs[node][1]
        batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[input_node][:bound], batch_info)
    elseif length(model_info.node_prevs[node]) == 0 #the node is start: use the pre-defined bound to start, and update the bound of the current node
        batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[node][:bound], batch_info)
    end
    return batch_bound
end

"""
    propagate_layer_method(prop_method::BackwardProp, model_info, batch_info, node)

This function propagates the bounds of the next node from the provided node 
using the specified forward propagation method and layer operation. It invokes 
`propagate_layer_batch`, which subsequently calls either 
`propagate_layer_batch` or `propagate_layer_batch`. The function identifies the 
next node from the given node in the computational graph, `model_info`, its 
bound, and the layer operation of the node. Then, `propagate_layer_batch` 
ascertains if the layer operation is linear or includes activation functions 
like ReLU. Depending on this, `propagate_layer_batch` or `propagate_layer_batch` 
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
    if length(model_info.node_nexts[node]) != 0
        output_node = model_info.node_nexts[node][1]
        batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[output_node][:bound], batch_info)
    else #the node is final_node: use the pre-defined bound to start, and update the bound of the current node
        batch_bound = propagate_layer_batch(prop_method, model_info.node_layer[node], batch_info[node][:bound], batch_info)
    end
    return batch_bound
end

"""
    propagate_layer_batch(prop_method::ForwardProp, layer, 
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
function propagate_layer_batch(prop_method::ForwardProp, layer, batch_reach::AbstractArray, batch_info)
    batch_reach_info = [propagate_layer(prop_method, layer, batch_reach[i], push!(batch_info, :batch_index => i)) for i in eachindex(batch_reach)]
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
    # fix bugs: there are no tanh and sigmod in NNlib.ACTIVATIONS
    isa(l, typeof(@eval NNlib.tanh)) && (l = tanh_fast)
    isa(l, typeof(@eval NNlib.sigmoid)) && (l = sigmoid_fast)
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
# function propagate_layer_batch(prop_method, layer, batch_bound, batch_info)
#     to = get_timer("Shared")
#     @timeit to "propagate_layer_batch" batch_bound = propagate_layer_batch(prop_method, layer, batch_bound, batch_info)
#     return batch_bound
# end


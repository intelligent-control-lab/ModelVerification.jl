abstract type AbstractNetwork end

"""
    Layer{F, N}

Consists of `weights` and `bias` for linear mapping, and `activation` for nonlinear mapping.
### Fields
 - `weights::Matrix{N}`
 - `bias::Vector{N}`
 - `activation::F`

See also: [`Network`](@ref)
"""
struct Layer{F<:ActivationFunction, N<:Number}
    weights::Matrix{N}
    bias::Vector{N}
    activation::F
end

"""
A Vector of layers.

    Network([layer1, layer2, layer3, ...])

See also: [`Layer`](@ref)
"""
struct Network <: AbstractNetwork
    layers::Vector{Layer} # layers includes output layer
end

"""
    n_nodes(L::Layer)

Returns the number of neurons in a layer.
"""
n_nodes(L::Layer) = length(L.bias)

function get_sub_model(model_info, end_node)
    # get the first part of the model until end_node.
    queue = Queue{Any}()
    enqueue!(queue, end_node)
    sub_nodes = [end_node]
    node_prevs = Dict(end_node => [])
    node_nexts = Dict(end_node => [])
    while !isempty(queue)
        node = dequeue!(queue)
        for input_node in model_info.node_prevs[node]
            if !(input_node in sub_nodes)
                push!(sub_nodes, input_node)
                enqueue!(queue, input_node)
                node_prevs[input_node] = []
                node_nexts[input_node] = []
            end
            push!(node_prevs[node], input_node)
            push!(node_nexts[input_node], node)
        end
    end
    node_layer = filter(kv -> kv[1] in sub_nodes, model_info.node_layer)
    start_nodes = filter(n -> n in sub_nodes, model_info.start_nodes)
    activation_nodes = filter(n -> n in sub_nodes, model_info.activation_nodes)
    activation_number = length(activation_nodes)
    return Model(start_nodes, [end_node], sub_nodes, node_layer, node_prevs, node_nexts, activation_nodes, activation_number)
end

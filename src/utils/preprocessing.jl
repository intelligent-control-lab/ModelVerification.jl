"""
    Model

Structure containing the information of the neural network to be verified.

## Fields
- start_nodes: List of input layer nodes' names.
- final_nodes: List of output layer nodes' names.
- all_nodes: List of all the nodes's names.
- node_layer: Dictionary of all the nodes. The key is the name of the node and 
    the value is the operation performed at the node.
- node_prevs: Dictionary of the nodes connected to the current node.
    The key is the name of the node and the value is the list of nodes.
- node_nexts: Dictionary of the nodes connected from the current node.
    The key is the name of the node and the value is the list of nodes.
- activation_nodes: List of all the activation nodes' names.
- activation_number: Number of activation nodes (deprecated in the future).
"""
struct Model
    start_nodes::Array{String, 1}
    final_nodes::Array{String, 1}
    all_nodes::Array{String, 1}
    node_layer::Dict
    node_prevs::Dict
    node_nexts::Dict
    activation_nodes::Array{String, 1}
    activation_number::Int
end

"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem)

Converts the given `Problem` into a form that is compatible with the verification
process of the toolbox. In particular, it retrieves information about the ONNX 
model to be verified and stores them into a `Model`. It returns the `Problem` 
itself and the `Model` structure. 

## Arguments
- `search_method` (`SearchMethod`): Search method for the verification process.
- `split_method` (`SplitMethod`): Split method for the verification process.
- `prop_method` (`PropMethod`): Propagation method for the verification process.
- `problem` (`Problem`): Problem definition for model verification.

## Returns
- `model_info` (`Model`): Information about the model to be verified.
- `problem` (`Problem`): The given problem definition for model verification.
"""
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, problem
end

"""
    get_act(l)

Returns the activation function of the node `l` if it exists.

## Arguments
- `l`: node

## Returns
- Activation function of the node if it exists.
- Otherwise, return `nothing`.
"""
function get_act(l)
    (hasfield(typeof(l), :σ) && string(l.σ) != "identity") && return l.σ
    (hasfield(typeof(l), :λ) && string(l.λ) != "identity") && return l.λ
    return nothing
end

"""
    onnx_parse(onnx_model_path)

Creates the `Model` from the `onnx_model_path`. First, the computational graph 
of the ONNX model is created. Then, the `Model` is created using the information
retrieved from the computational graph.

## Arguments
- `onnx_model_path`: String path to the ONNX model.

## Returns
- `model_info` (`Model`): Contains network information retrieved from the 
    computational graph of the ONNX model.
"""
function onnx_parse(onnx_model_path)
    @assert !isnothing(onnx_model_path) 
    comp_graph = ONNXNaiveNASflux.load(onnx_model_path, infer_shapes=false)
    activation_number = 0
    start_nodes = []
    activation_nodes = []
    final_nodes = []
    all_nodes = []
    node_prevs = Dict()
    node_nexts = Dict()
    node_layer = Dict()
    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        
        node_name = NaiveNASflux.name(vertex)
        if length(inputs(vertex)) == 0 # the start node has no input nodes
            push!(start_nodes, node_name)
        end
        # println("NaiveNASflux.layer(vertex)")
        # println(NaiveNASflux.layer(vertex))
        if length(string(NaiveNASflux.name(vertex))) >= 7 && string(NaiveNASflux.name(vertex))[1:7] == "Flatten" 
            node_layer[node_name] = Flux.flatten
        elseif length(string(NaiveNASflux.name(vertex))) >= 3 && string(NaiveNASflux.name(vertex))[1:3] == "add" 
            node_layer[node_name] = +
        elseif length(string(NaiveNASflux.name(vertex))) >= 4 && string(NaiveNASflux.name(vertex))[1:4] == "relu" 
            # activation_number += 1
            # node_name = "relu" * "_" * string(activation_number) #activate == "relu_5" doesn't mean this node is 5th relu node, but means this node is 5th activation node
            node_layer[node_name] = NNlib.relu
            push!(activation_nodes, node_name)
        else
            node_layer[node_name] = NaiveNASflux.layer(vertex)
        end

        push!(all_nodes, node_name)

        node_nexts[node_name] = [NaiveNASflux.name(output_node) for output_node in outputs(vertex)]

        # add input nodes of current node. If the input nodes of current node have activation(except identity), then the "inputs" should be the activation node
        node_prevs[node_name] = []
        # if !(node_name in start_nodes)# if current node is not one of the start node
        for input_node in inputs(vertex)
            input_node_name = NaiveNASflux.name(input_node)
            # println("---")
            # println(input_node_name)
            # println(node_layer[input_node_name])
            act = get_act(node_layer[input_node_name])
            if !isnothing(act)
                prev_activation_name = input_node_name * "_" * string(act)
                push!(node_prevs[node_name], prev_activation_name)    
            else
                push!(node_prevs[node_name], input_node_name)
            end
        end
        # end
        # println(typeof(NaiveNASflux.layer(vertex)))
        # println(hasfield(typeof(NaiveNASflux.layer(vertex)), :σ))
        # println(hasfield(typeof(NaiveNASflux.layer(vertex)), :λ))
        #split this layer into a linear layer and a activative layer
        act = get_act(NaiveNASflux.layer(vertex))
        if !isnothing(act)
            activation_name = node_name * "_" * string(act)
            
            node_prevs[activation_name] = [node_name]
            node_nexts[activation_name] = node_nexts[node_name]

            node_nexts[node_name] = [activation_name]

            node_layer[activation_name] = act
            
            push!(activation_nodes, activation_name)
            push!(all_nodes, activation_name) 
            node_name = activation_name  # for getting the final_nodes
        end
        
        if length(outputs(vertex)) == 0  #the final node has no output nodes
            push!(final_nodes, node_name) 
        end
        # println(node_name)
        # println("prevs: ", node_prevs[node_name])
        # println("nexts: ", node_nexts[node_name])
        # println("====")
    end
    model_info = Model(start_nodes, final_nodes, all_nodes, node_layer, node_prevs, node_nexts, activation_nodes, activation_number)
    # println("model_info.start_nodes")
    # println(model_info.start_nodes)
    return model_info

end
 

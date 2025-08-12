"""
    Model

Structure containing the information of the neural network to be verified.

## Fields
- `start_nodes` (`Array{String, 1}`): List of input layer nodes' names.
- `final_nodes` (`Array{String, 1}`): List of output layer nodes' names.
- `all_nodes` (`Array{String, 1}`): List of all the nodes's names.
- `node_layer` (`Dict`): Dictionary of all the nodes. The key is the name of the 
    node and the value is the operation performed at the node.
- `node_prevs` (`Dict`): Dictionary of the nodes connected to the current node.
    The key is the name of the node and the value is the list of nodes. (actually the values are the names of the nodes)
- `node_nexts` (`Dict`): Dictionary of the nodes connected from the current 
    node. The key is the name of the node and the value is the list of nodes. (actually the values are the names of the nodes)
- `activation_nodes` (`Array{String, 1}`): List of all the activation nodes' 
    names.
- `activation_number` (`Int`): Number of activation nodes (deprecated in the 
    future).
"""
struct ModelGraph
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
    convert(::Type{ModelVerification.ModelGraph}, model::OnnxNet)

Converts from VNNLib.OnnxParser's `OnnxNet` to `ModelVerification.ModelGraph`.

## Arguments
- `model`: An instance of `OnnxNet` from VNNLib.OnnxParser

## Returns
- `ModelVerification.ModelGraph`: A model graph representation of the ONNX model.
"""
function Base.convert(::Type{ModelGraph}, model::OnnxNet)
    # ModelVerification needs a separate node to handle input variables.
    model = add_dummy_input_node(model)

    # Convert the ONNX model to a ModelGraph
    start_nodes = model.start_nodes
    final_nodes = model.final_nodes
    all_nodes = collect(keys(model.nodes))
    # node_layer = Dict(name => onnx_node_to_flux_layer(node) for (name, node) in model.nodes)
    # TODO: this is maybe the greatest change here.
    #       Don't directly use Flux layers, but OnnxParser intermediate representation.
    #       This is necessary to avoid the problem that the Flux layers are not
    #       compatible with the ONNX model.
    node_layer = model.nodes
    node_prevs = model.node_prevs
    node_nexts = model.node_nexts
    activation_nodes = [name for (name, node) in model.nodes if node isa OXP.ONNXRelu]
    activation_number = length(activation_nodes)

    return ModelGraph(start_nodes, final_nodes, all_nodes, node_layer, node_prevs, node_nexts, activation_nodes, activation_number)
end


function onnx_parse(onnx_path)
    comp_graph = load_onnx_model(onnx_path)
    model_info = convert(ModelGraph, comp_graph)
    return model_info
end


"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, 
                    prop_method::PropMethod, problem::Problem)

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



function parent_nodes(comp_graph::OnnxNet, vertex::OXP.Node)
    parents = [comp_graph.nodes[name] for name in comp_graph.node_prevs[vertex.name]]
    return parents
end

function next_nodes(comp_graph::OnnxNet, vertex::OXP.Node)
    nexts = [comp_graph.nodes[name] for name in comp_graph.node_nexts[vertex.name]]
    return nexts
end


function compute_output(model_info, batch_input::AbstractArray)

    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.start_nodes[1]][:out] = batch_input
    # return compute_output(model_info, batch_info)

    queue = Queue{Any}()
    # @show [model_info.node_nexts[s] for s in model_info.start_nodes]
    # @show vcat([model_info.node_nexts[s] for s in model_info.start_nodes]...)
    # foreach(x -> enqueue!(queue, x), vcat([model_info.node_nexts[s] for s in model_info.start_nodes]...))
    foreach(x -> enqueue!(queue, x), model_info.start_nodes)

    out_cnt = Dict(node => 0 for node in model_info.all_nodes)
    visit_cnt = Dict(node => 0 for node in model_info.all_nodes)
    i = 0

    SNRs = []
    out_and_bounds = Dict()

    while !isempty(queue)
        i += 1
        node = dequeue!(queue)
        # @show node, typeof(model_info.node_layer[node])
        batch_info[:current_node] = node
        for output_node in model_info.node_nexts[node]
            visit_cnt[output_node] += 1
            if visit_cnt[output_node] == length(model_info.node_prevs[output_node])
                enqueue!(queue, output_node)
            end
        end
        
        node in model_info.start_nodes && continue # start nodes do not need computing bound

        if length(model_info.node_prevs[node]) == 2
            batch_out = compute_out_skip(model_info, batch_info, node)
        else
            batch_out = compute_out_layer(model_info, batch_info, node)
        end
        
        batch_info[node][:out] = batch_out
    end
    
    return batch_info
end

function compute_out_skip(model_info, batch_info, node)
    input_node1 = model_info.node_prevs[node][1]
    input_node2 = model_info.node_prevs[node][2]
    batch_out1 = haskey(batch_info[input_node1], :out) ? batch_info[input_node1][:out] : get_center(batch_info[input_node1][:bound][1])
    batch_out2 = haskey(batch_info[input_node2], :out) ? batch_info[input_node2][:out] : get_center(batch_info[input_node2][:bound][1])
    return model_info.node_layer[node](batch_out1 |> cpu, batch_out2 |> cpu)
end

function compute_out_layer(model_info, batch_info, node)
    input_node1 = model_info.node_prevs[node][1]
    batch_out1 = batch_info[input_node1][:out]
    return model_info.node_layer[node](batch_out1 |> cpu)
end

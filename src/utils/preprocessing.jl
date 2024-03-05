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
    The key is the name of the node and the value is the list of nodes.
- `node_nexts` (`Dict`): Dictionary of the nodes connected from the current 
    node. The key is the name of the node and the value is the list of nodes.
- `activation_nodes` (`Array{String, 1}`): List of all the activation nodes' 
    names.
- `activation_number` (`Int`): Number of activation nodes (deprecated in the 
    future).
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

function onnx_node_to_flux_layer(vertex)
    node_type = NaiveNASflux.layertype(vertex)
    # println("node_type: ",node_type)
    # println(ONNXNaiveNASflux.var"#342#344"{typeof(relu)})
    if node_type == ONNXNaiveNASflux.Flatten
        return Flux.flatten
    elseif node_type == NaiveNASlib.var"#342#344"{typeof(+)}
        return +
    elseif node_type == NaiveNASlib.var"#342#344"{typeof(-)}
        return -
    # elseif node_type == ONNXNaiveNASflux.var"#217#229"{typeof(relu)}
    #     return NNlib.relu
    else
        # @show vertex
        return NaiveNASflux.layer(vertex)
    end
end

function rename_same_name(comp_graph)
    layer_cnt = Dict() # used for naming
    node_name = Dict() 
    all_names = Set()
    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        node_type = nameof(typeof(NaiveNASflux.layer(vertex)))
        layer_cnt[node_type] = haskey(layer_cnt, node_type) ? layer_cnt[node_type]+1 : 1
        
        v_name = NaiveNASflux.name(vertex)
        if v_name == "" || v_name in all_names # has the same name as other nodes, needs renaming
            v_name = string(node_type) * "_" * string(layer_cnt[node_type])
        end
        push!(all_names, v_name)
        node_name[vertex] = v_name
    end
    return node_name
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
    act_name = Dict()
    
    node_name = rename_same_name(comp_graph) # some nodes can have the same name, need to distinguish them

    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        
        v_name = node_name[vertex]

        if length(inputs(vertex)) == 0 # the start node has no input nodes
            push!(start_nodes, v_name)
            node_layer[v_name] = identity # data node only exists in onnx, therefore we assign it an identity operator for flux model.
        else
            node_layer[v_name] = onnx_node_to_flux_layer(vertex)
        end

        # println("NaiveNASflux.layer(vertex) === ")
        # println(v_name)
        # println(typeof(vertex))
        # println(NaiveNASflux.layertype(vertex))
        # println(NaiveNASflux.layer(vertex))
        # println(typeof(NaiveNASflux.layer(vertex)))
        
        
        # println("name === layer")
        # println(v_name)
        # println(node_layer[v_name])
        push!(all_nodes, v_name)

        if isa(node_layer[v_name], typeof(NNlib.relu))
            push!(activation_nodes, v_name)
        end

        node_nexts[v_name] = [node_name[output_node] for output_node in outputs(vertex)]

        # add input nodes of current node. If the input nodes of current node have activation(except identity), then the "inputs" should be the activation node
        node_prevs[v_name] = []
        for input_node in inputs(vertex)
            input_node_name = node_name[input_node]
            if haskey(act_name, input_node_name)
                push!(node_prevs[v_name], act_name[input_node_name])
            else
                push!(node_prevs[v_name], input_node_name)
            end
        end
        
        #split this layer into a linear layer and a activative layer
        act = get_act(NaiveNASflux.layer(vertex))
        if !isnothing(act)
            act_name[v_name] = v_name * "_" * string(act)

            node_prevs[act_name[v_name]] = [v_name]
            node_nexts[act_name[v_name]] = node_nexts[v_name]

            node_nexts[v_name] = [act_name[v_name]]
            node_layer[act_name[v_name]] = act

            push!(activation_nodes, act_name[v_name])
            push!(all_nodes, act_name[v_name]) 

            node_layer[v_name] = remove_layer_act(node_layer[v_name])

            v_name = act_name[v_name]  # for getting the final_nodes
        end
        
        if length(outputs(vertex)) == 0  #the final node has no output nodes
            push!(final_nodes, v_name) 
        end
        # println(node_name)
        # println("prevs: ", node_prevs[node_name])
        # println("nexts: ", node_nexts[node_name])
        # println("====")
    end

    # println("node_layer")
    # println(node_layer)
    # @assert false
    model_info = Model(start_nodes, final_nodes, all_nodes, node_layer, node_prevs, node_nexts, activation_nodes, activation_number)
    # println("model_info.start_nodes")
    # println(model_info.start_nodes)
    model_info = purify_model(model_info) # some ad-hoc treatment for vnncomp benchmarks models
    return model_info

end

function remove_layer_act(l)
    if l isa Dense
        return Dense(l.weight, l.bias, identity;)
    elseif l isa Conv
        return Conv(l.weight, l.bias, identity, stride = l.stride, pad = l.pad, dilation = l.dilation, groups = l.groups)
    elseif l isa ConvTranspose
        return ConvTranspose(l.weight, l.bias, identity, stride = l.stride, pad = l.pad, dilation = l.dilation, groups = l.groups)
    else
        @warn "Decoupling activation for $l is not implemented. The inference output may be incorrect. Verification is not affected."
    end
end

function remove_start_flatten(model_info::Model)
    length(model_info.start_nodes) > 1 && return model_info
    node1 = model_info.start_nodes[1] 
    length(model_info.node_nexts[node1]) > 1 && return model_info
    node2 = model_info.node_nexts[node1][1]
    model_info.node_layer[node2] != Flux.flatten && return model_info
    node3 = model_info.node_nexts[node2][1]
    model_info.node_nexts[node1] = model_info.node_nexts[node2]
    model_info.node_prevs[node3] = model_info.node_prevs[node2]
    return model_info
end

# some ad-hoc treatment for vnncomp benchmarks models
function purify_model(model_info::Model) 
    model_info = remove_start_flatten(model_info)
end
 

function compute_output(model_info, batch_input::AbstractArray)

    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.start_nodes[1]][:out] = batch_input
    # return compute_output(model_info, batch_info)
# end

# function compute_output(model_info, batch_info::Dict)
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
    batch_out1 = haskey(batch_info[input_node1], :out) ? batch_info[input_node1][:out] : center(batch_info[input_node1][:bound][1])
    batch_out2 = haskey(batch_info[input_node2], :out) ? batch_info[input_node2][:out] : center(batch_info[input_node2][:bound][1])
    return model_info.node_layer[node](batch_out1 |> cpu, batch_out2 |> cpu)
end

function compute_out_layer(model_info, batch_info, node)
    input_node1 = model_info.node_prevs[node][1]
    batch_out1 = batch_info[input_node1][:out]
    return model_info.node_layer[node](batch_out1 |> cpu)
end

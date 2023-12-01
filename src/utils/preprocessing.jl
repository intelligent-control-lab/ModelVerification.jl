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

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, problem
end

function get_act(l)
    (hasfield(typeof(l), :σ) && string(l.σ) != "identity") && return l.σ
    (hasfield(typeof(l), :λ) && string(l.λ) != "identity") && return l.λ
    return nothing
end

# function onnx_node_to_flux_layer(vertex)
#     node_name = NaiveNASflux.name(vertex)
#     if occursin("flatten", lowercase(node_name))
#         return Flux.flatten
#     elseif occursin("add", lowercase(node_name))
#         return +
#     elseif occursin("sub", lowercase(node_name))
#         return -
#     elseif occursin("relu", lowercase(node_name))
#         # activation_number += 1
#         # node_name = "relu" * "_" * string(activation_number) #activate == "relu_5" doesn't mean this node is 5th relu node, but means this node is 5th activation node
#         return NNlib.relu
#     else
#         return NaiveNASflux.layer(vertex)
#     end
# end
function onnx_node_to_flux_layer(vertex)
    node_type = NaiveNASflux.layertype(vertex)
    # println("node_type: ",node_type)
    if node_type == ONNXNaiveNASflux.Flatten
        return Flux.flatten
    elseif node_type == NaiveNASlib.var"#342#344"{typeof(+)}
        return +
    elseif node_type == NaiveNASlib.var"#342#344"{typeof(-)}
        return -
    elseif node_type == ONNXNaiveNASflux.var"#217#229"{typeof(relu)}
        return NNlib.relu
    else
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
    
    node_name = rename_same_name(comp_graph) # some nodes can have the same name, need to distinguish them

    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        
        v_name = node_name[vertex]

        if length(inputs(vertex)) == 0 # the start node has no input nodes
            push!(start_nodes, v_name)
        end

        # println("NaiveNASflux.layer(vertex) === ")
        # println(v_name)
        # println(typeof(vertex))
        # println(NaiveNASflux.layertype(vertex))
        # println(NaiveNASflux.layer(vertex))
        # println(typeof(NaiveNASflux.layer(vertex)))
        
        node_layer[v_name] = onnx_node_to_flux_layer(vertex)
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
            act = get_act(node_layer[input_node_name])
            if !isnothing(act)
                prev_activation_name = input_node_name * "_" * string(act)
                push!(node_prevs[v_name], prev_activation_name)    
            else
                push!(node_prevs[v_name], input_node_name)
            end
        end
        
        #split this layer into a linear layer and a activative layer
        act = get_act(NaiveNASflux.layer(vertex))
        if !isnothing(act)
            activation_name = v_name * "_" * string(act)
            
            node_prevs[activation_name] = [v_name]
            node_nexts[activation_name] = node_nexts[v_name]

            node_nexts[v_name] = [activation_name]
            node_layer[activation_name] = act
            
            push!(activation_nodes, activation_name)
            push!(all_nodes, activation_name) 
            v_name = activation_name  # for getting the final_nodes
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

function purify_model(model_info::Model)
    model_info = remove_start_flatten(model_info)
end
 

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

function onnx_node_to_flux_layer(vertex)
    node_name = NaiveNASflux.name(vertex)
    if occursin("flatten", lowercase(node_name))
        return Flux.flatten
    elseif occursin("add", lowercase(node_name))
        return +
    elseif occursin("sub", lowercase(node_name))
        return -
    elseif occursin("relu", lowercase(node_name))
        # activation_number += 1
        # node_name = "relu" * "_" * string(activation_number) #activate == "relu_5" doesn't mean this node is 5th relu node, but means this node is 5th activation node
        return NNlib.relu
    else
        return NaiveNASflux.layer(vertex)
    end
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
    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        
        node_name = NaiveNASflux.name(vertex)
        # node_name = lowercase(node_name)

        if length(inputs(vertex)) == 0 # the start node has no input nodes
            push!(start_nodes, node_name)
        end

        # println("NaiveNASflux.layer(vertex)")
        # println(node_name)
        # println(typeof(vertex))
        
        node_layer[node_name] = onnx_node_to_flux_layer(vertex)
        
        push!(all_nodes, node_name)
        if isa(node_layer[node_name], typeof(NNlib.relu))
            push!(activation_nodes, node_name)
        end

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
    # println("node_layer")
    # println(node_layer)
    # @assert false
    model_info = Model(start_nodes, final_nodes, all_nodes, node_layer, node_prevs, node_nexts, activation_nodes, activation_number)
    # println("model_info.start_nodes")
    # println(model_info.start_nodes)
    return model_info

end
 

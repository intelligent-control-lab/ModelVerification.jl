struct Model 
    start_nodes::Array{String, 1}
    final_nodes::Array{String, 1}
    all_nodes::Array{String, 1}
    activation_nodes::Array{String, 1}
    activation_number::Int
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem)
    batch_info, model_info = onnx_parse(problem.onnx_model_path, problem.Flux_model, size(problem.input))
    return batch_info, model_info, problem
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageStar, problem::Problem)
    batch_info, model_info = onnx_parse(problem.onnx_model_path, problem.Flux_model, size(problem.input))
    return batch_info, model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
end

function onnx_parse(onnx_model_path, Flux_model, input_shape)
    @assert !isnothing(onnx_model_path) 

    if !isnothing(Flux_model) 
        save(onnx_model_path, Flux_model, input_shape)
    end  

    comp_graph = ONNXNaiveNASflux.load(onnx_model_path, infer_shapes=false)
    batch_info = Dict() # store the information of node
    activation_number = 0
    start_nodes = []
    activation_nodes = []
    final_nodes = []
    all_nodes = []
    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        if index == 1 # the vertex which index == 1 has no useful information, so it's output node will be the start node of the model
            start_nodes = [NaiveNASflux.name(output_node) for output_node in outputs(vertex)]
            continue
        end 
        node_name = NaiveNASflux.name(vertex)
        new_dict = Dict() # store the information of this vertex 
        push!(new_dict, "vertex" => vertex)
        push!(new_dict, "layer" => NaiveNASflux.layer(vertex))
        push!(new_dict, "index" => index)
        push!(new_dict, "outputs" => [NaiveNASflux.name(output_node) for output_node in outputs(vertex)])
        # add input nodes of current node. If the input nodes of current node have activation(except identity), then the "inputs" should be the activation node
        if !(node_name in start_nodes)# if current node is not one of the start node
            push!(new_dict, "inputs" => [])
            for input_node in inputs(vertex)
                input_node_name = NaiveNASflux.name(input_node)
                if hasfield(typeof(batch_info[input_node_name]["layer"]), :σ) && string(batch_info[input_node_name]["layer"].σ) != "identity"
                    push!(new_dict["inputs"], batch_info[input_node_name]["outputs"][1])
                else
                    push!(new_dict["inputs"], input_node_name)
                end
            end
        else
            push!(new_dict, "inputs" => nothing)
        end
        
        if length(string(NaiveNASflux.name(vertex))) >= 7 && string(NaiveNASflux.name(vertex))[1:7] == "Flatten" 
            push!(new_dict, "layer" => Flux.flatten)
            push!(batch_info, node_name => new_dict) #new_dict belongs to batch_info
            push!(all_nodes, node_name) 
        elseif length(string(NaiveNASflux.name(vertex))) >= 3 && string(NaiveNASflux.name(vertex))[1:3] == "add" 
            push!(new_dict, "layer" => +)
            push!(batch_info, node_name => new_dict) #new_dict belongs to batch_info
            push!(all_nodes, node_name) 
        elseif length(string(NaiveNASflux.name(vertex))) >= 4 && string(NaiveNASflux.name(vertex))[1:4] == "relu" 
            activation_number += 1
            node_name = "relu" * "_" * string(activation_number) #activate == "relu_5" doesn't mean this node is 5th relu node, but means this node is 5th activation node
            push!(new_dict, "layer" => NNlib.relu)
            push!(batch_info, node_name => new_dict) #new_dict belongs to batch_info
            push!(activation_nodes, node_name)
            push!(all_nodes, node_name) 
        elseif hasfield(typeof(NaiveNASflux.layer(vertex)), :σ) && string(NaiveNASflux.layer(vertex).σ) != "identity"#split this layer into a linear layer and a activative layer
            activation_number += 1
            activation_name = string(NaiveNASflux.layer(vertex).σ) * "_" * string(activation_number)
            push!(new_dict, "outputs" => [activation_name]) #new_dict store the information of the linear layer
            push!(batch_info, node_name => new_dict) #new_dict belongs to batch_info
            push!(all_nodes, node_name) 
                
            activation_new_dict = Dict()#store the information of the activative layer
            push!(activation_new_dict, "vertex" => vertex)
            push!(activation_new_dict, "layer" => NaiveNASflux.layer(vertex).σ)
            push!(activation_new_dict, "index" => index)# Do not need to change index
            push!(activation_new_dict, "inputs" => [node_name])
            push!(activation_new_dict, "outputs" => [NaiveNASflux.name(output_nodes) for output_nodes in outputs(vertex)])
            push!(batch_info, activation_name => activation_new_dict)
            push!(activation_nodes, activation_name)
            push!(all_nodes, activation_name) 

            node_name = activation_name #for getting the final_nodes
        else
            push!(batch_info, node_name => new_dict) #new_dict belongs to batch_info
            push!(all_nodes, node_name) 
        end
        
        if length(batch_info[node_name]["outputs"]) == 0  #the final node has no output nodes
            push!(final_nodes, node_name) 
            push!(batch_info[node_name], "outputs" => nothing) 
        end
    end
    model_info = Model(start_nodes, final_nodes, all_nodes, activation_nodes, activation_number)
    return batch_info, model_info

end
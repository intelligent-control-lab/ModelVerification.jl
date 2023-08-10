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

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::Crown, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    if prop_method.use_gpu
        return model_info, Problem(problem.onnx_model_path, fmap(cu, problem.Flux_model), fmap(cu, prop_method, problem.input), fmap(cu, problem.output))
    else
        return model_info, problem
    end
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::AlphaCrown, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    if prop_method.use_gpu
        return model_info, Problem(problem.onnx_model_path, fmap(cu, problem.Flux_model), fmap(cu, problem.input), fmap(cu, problem.output))
    else
        return model_info, problem
    end
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::BetaCrown, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    if prop_method.use_gpu
        return model_info, Problem(problem.onnx_model_path, fmap(cu, problem.Flux_model), fmap(cu, init_bound(prop_method, problem.input)), fmap(cu, problem.output))
    else
        return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
    end
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageStar, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
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
        
        if length(inputs(vertex)) == 0 # the start node has no input nodes
            push!(start_nodes, node_name)
        end
        
        if length(string(NaiveNASflux.name(vertex))) >= 7 && string(NaiveNASflux.name(vertex))[1:7] == "Flatten" 
            node_layer[node_name] = Flux.flatten
        elseif length(string(NaiveNASflux.name(vertex))) >= 3 && string(NaiveNASflux.name(vertex))[1:3] == "add" 
            node_layer[node_name] = +
        elseif length(string(NaiveNASflux.name(vertex))) >= 4 && string(NaiveNASflux.name(vertex))[1:4] == "relu" 
            activation_number += 1
            node_name = "relu" * "_" * string(activation_number) #activate == "relu_5" doesn't mean this node is 5th relu node, but means this node is 5th activation node
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
            if hasfield(typeof(node_layer[input_node_name]), :σ) && string(node_layer[input_node_name].σ) != "identity"
                push!(node_prevs[node_name], node_nexts[input_node_name][1])
            else
                push!(node_prevs[node_name], input_node_name)
            end
        end
        # end
        
        if hasfield(typeof(NaiveNASflux.layer(vertex)), :σ) && string(NaiveNASflux.layer(vertex).σ) != "identity"#split this layer into a linear layer and a activative layer
            activation_number += 1
            activation_name = string(NaiveNASflux.layer(vertex).σ) * "_" * string(activation_number)
            
            node_prevs[activation_name] = [node_name]
            node_nexts[activation_name] = node_nexts[node_name]

            node_nexts[node_name] = [activation_name]

            node_layer[activation_name] = NaiveNASflux.layer(vertex).σ
            
            push!(activation_nodes, activation_name)
            push!(all_nodes, activation_name) 
            node_name = activation_name  # for getting the final_nodes
        end
        
        if length(outputs(vertex)) == 0  #the final node has no output nodes
            push!(final_nodes, node_name) 
        end
    end
    model_info = Model(start_nodes, final_nodes, all_nodes, node_layer, node_prevs, node_nexts, activation_nodes, activation_number)
    return model_info

end

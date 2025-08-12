
using ModelVerification, Flux, VNNLib, LazySets, TimerOutputs
import VNNLib.OnnxParser: onnx_node_to_flux_layer
const OXP = VNNLib.OnnxParser
const MV = ModelVerification

## Convert OnnxNet to Flux

function parent_nodes(comp_graph::OnnxNet, vertex::VNNLib.OnnxParser.Node)
    parents = [comp_graph.nodes[name] for name in comp_graph.node_prevs[vertex.name]]
    return parents
end

function next_nodes(comp_graph::OnnxNet, vertex::VNNLib.OnnxParser.Node)
    nexts = [comp_graph.nodes[name] for name in comp_graph.node_nexts[vertex.name]]
    return nexts
end

"""
    my_get_chain(vertex)

Returns a `Flux.Chain` constructed from the given vertex. This is a helper 
function for `build_flux_model`. 

## Arguments
- `vertex`: Vertex from the `NaiveNASflux` computation graph.

## Returns
- `model`: `Flux.Chain` constructed from the given vertex.
- `curr_vertex`: The last vertex in the chain.
"""
function my_get_chain(model::OnnxNet, vertex::VNNLib.OnnxParser.Node)
    m = Any[]
    curr_vertex = vertex
    # println("getting chain start from:", NaiveNASflux.name(curr_vertex))
    
    # while the current node is not the merging node of a parallel layer
    while length(curr_vertex.inputs) < 2
        # println("push:", NaiveNASflux.name(curr_vertex))
        # push!(m, NaiveNASflux.layer(curr_vertex))
        push!(m, onnx_node_to_flux_layer(curr_vertex))

        outs = next_nodes(model, curr_vertex)
        while length(outs) == 2
            chain1, end_node1 = my_get_chain(model, outs[1])
            chain2, end_node2 = my_get_chain(model, outs[2])
            @assert end_node1 == end_node2
            op = onnx_node_to_flux_layer(end_node1)
            if length(chain1) == 0
                push!(m, SkipConnection(chain2, op))
            elseif length(chain2) == 0
                push!(m, SkipConnection(chain1, op))
            else
                push!(m, Parallel(op; α = chain1, β = chain2))
            end
            # curr_vertex = NaiveNASflux.outputs(end_node1)[1]
            curr_vertex = end_node1
            # println("merging chain:", NaiveNASflux.name(curr_vertex))
        end
        outs = next_nodes(model, curr_vertex)
        length(outs) == 0 && break
        curr_vertex = outs[1]
    end
    return Chain(m...), curr_vertex
end

"""
    build_flux_model(onnx_model_path)

Builds a `Flux.Chain` from the given ONNX model path.

## Arguments
- `onnx_model_path`: String path to ONNX model in `.onnx` file.

## Returns
- `model`: `Flux.Chain` constructed from the `.onnx` file.
"""
function build_flux_model(onnx_model_path)
    comp_graph = load_onnx_model(onnx_model_path)
    start_vertex = [comp_graph.nodes[vertex_name] for vertex_name in comp_graph.start_nodes]
    @assert length(start_vertex) == 1 "Currently only one start vertex is supported, found $(length(start_vertex))"
    model_vec, end_node = my_get_chain(comp_graph, start_vertex[1])
    model = Chain(model_vec...)
    # model = purify_flux_model(model)
    return model
end


## Don't try to convert Flux model to ONNX again

# Just add thing_to_match, s.t. we don't call the standard constructor
ModelVerification.Problem(model::Chain, input_data, output_data, path::String) = #If the Problem only have onnx model input
    Problem(path, model, input_data, output_data)


## Convert OnnxNet to ModelGraph

function Base.convert(::Type{ModelVerification.ModelGraph}, model::OnnxNet)
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

    return ModelVerification.ModelGraph(start_nodes, final_nodes, all_nodes, node_layer, node_prevs, node_nexts, activation_nodes, activation_number)
end

function my_prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem)
    comp_graph = load_onnx_model(problem.onnx_model_path)
    model_info = convert(ModelVerification.ModelGraph, comp_graph)
    return model_info, problem 
end


## Adjust propagation methods to work with ONNX layers

function MV.propagate_layer_batch(prop_method::Crown, node::OXP.ONNXLinear, bound::MV.CrownBound, batch_info)
    @assert node.transpose == false "Transpose argument is currently not supported for ONNXLinear!"
    MV.propagate_layer_batch(prop_method, node.dense, bound, batch_info)
end


function MV.propagate_layer_batch(prop_method::Crown, node::OXP.ONNXRelu, original_bound::MV.CrownBound, batch_info)
    MV.propagate_layer_batch(prop_method, Flux.relu, original_bound, batch_info)
end


## Just copy the verify method and swap the functions to the ones we defined above.

function my_verify(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem; 
                time_out=86400, 
                attack_restart=100, 
                collect_bound=false, 
                summary=false, 
                pre_split=nothing,
                search_adv_bound=false,
                comp_verified_ratio=false,
                verbose=false)
    to = get_timer("Shared")
    reset_timer!(to)
    # @timeit to "attack" res = attack(problem; restart=attack_restart)
    # (res.status == :violated) && (return res)
    @timeit to "prepare_problem" model_info, prepared_problem = my_prepare_problem(search_method, split_method, prop_method, problem)
    # println(time_out)   
    @timeit to "search_branches" res, verified_bounds, verified_ratio = ModelVerification.search_branches(search_method, split_method, prop_method, prepared_problem, model_info, collect_bound=collect_bound, comp_verified_ratio=comp_verified_ratio, pre_split=pre_split, verbose=verbose)
    # println(res.status)
    info = Dict()
    (res.status == :violated && res isa CounterExampleResult) && (info[:counter_example] = res.counter_example)
    collect_bound && (info[:verified_bounds] = verified_bounds)
    comp_verified_ratio && (info[:verified_ratio] = verified_ratio)
    
    if res.status != :holds && search_adv_bound
        info[:adv_input_scale], info[:adv_input_bound] = ModelVerification.search_adv_input_bound(search_method, split_method, prop_method, problem)# unknown or violated
    end
    summary && show(to) # to is TimerOutput(), used to profiling the code
    return MV.ResultInfo(res.status, info)
end


## Introductory example from Readme


# load model
onnx_path = "models/small_nnet.onnx"
toy_model = build_flux_model(onnx_path)

# define property
X = Hyperrectangle(low = [-2.5], high = [2.5])
Y = Hyperrectangle(low = [18.5], high = [114.5])

problem = Problem(toy_model, X, Y, onnx_path)

# define solver parameters
search_method = BFS(max_iter=100, batch_size=1)
split_method = Bisect(1)

# TODO: why did they only use zero_slope as Crown heuristic?
solver = Crown(use_gpu=false, bound_lower=true, bound_upper=true)

# solve the problem
result = my_verify(search_method, split_method, solver, problem)
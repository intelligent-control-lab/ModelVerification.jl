"""
    get_chain(vertex)
"""
function get_chain(vertex)
    m = Any[]
    curr_vertex = vertex
    # println("getting chain start from:", NaiveNASflux.name(curr_vertex))
    
    # while the current node is not the merging node of a parallel layer
    while length(NaiveNASflux.inputs(curr_vertex)) < 2
        # println("push:", NaiveNASflux.name(curr_vertex))
        push!(m, NaiveNASflux.layer(curr_vertex))

        while length(NaiveNASflux.outputs(curr_vertex)) == 2
            chain1, end_node1 = get_chain(NaiveNASflux.outputs(curr_vertex)[1])
            chain2, end_node2 = get_chain(NaiveNASflux.outputs(curr_vertex)[2])
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
        length(NaiveNASflux.outputs(curr_vertex)) == 0 && break
        curr_vertex = NaiveNASflux.outputs(curr_vertex)[1]
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
    comp_graph = ONNXNaiveNASflux.load(onnx_model_path, infer_shapes=false)
    model_vec = Any[]
    start_vertex = [vertex for vertex in ONNXNaiveNASflux.vertices(comp_graph) if isa(vertex, NaiveNASflux.InputShapeVertex)]
    @assert length(start_vertex) == 1
    @assert length(NaiveNASflux.outputs(start_vertex[1])) == 1
    model_vec, end_node = get_chain(NaiveNASflux.outputs(start_vertex[1])[1])
    model = Chain(model_vec...)
    model = purify_flux_model(model)
    return model
end


function remove_flux_start_flatten(model::Chain)
    !isa(model[1], ONNXNaiveNASflux.Flatten) && return model
    println("removing starting flatten")
    return model[2:end]
end
function purify_flux_model(model::Chain)
    model = remove_flux_start_flatten(model)
end
 


get_shape(input::ImageConvexHull) = (size(input.imgs[1])..., length(input.imgs))
get_shape(input::Hyperrectangle) = (size(LazySets.center(input))..., 1)
function build_onnx_model(path, model::Chain, input::InputSpec)
    ONNXNaiveNASflux.save(path, model, get_shape(input))
    return path
end

"""
    Problem{P, Q}(network::Network, input::P, output::Q)

Problem definition for neural verification.
The verification problem consists of: for all  points in the input set,
the corresponding output of the network must belong to the output set.

There are three ways to construct a `Problem`:
1. `Problem(path::String, model::Chain, input_data, output_data)` if both the  
    `.onnx` model path and `Flux_model` are given.
2. `Problem(path::String, input_data, output_data)` if only the `.onnx` model 
    path is given.
3. `Problem(model::Chain, input_data, output_data)` if only the `Flux_model` is 
    given.

## Fields
- `network` : `Network` that can be constructed either using the path to an onnx
    model or a `Flux.Chain` structure.
- `input` : input specification defined using a LazySet.
- `output` : output specification defined using a LazySet.
"""
struct Problem{P, Q}
    onnx_model_path::String
    Flux_model::Chain
    input::P
    output::Q
end
Problem(path::String, input_data, output_data) = #If the Problem only have onnx model input
    Problem(path, build_flux_model(path), input_data, output_data)
Problem(model::Chain, input_data, output_data; save_onnx_path="tmp.onnx") = #If the Problem only have Flux_model input
    Problem(build_onnx_model(save_onnx_path, model, input_data), model, input_data, output_data)

"""
    Result
    
Supertype of all result types.

See also: 
- [`BasicResult`](@ref) 
- [`CounterExampleResult`](@ref)
- [`AdversarialResult`](@ref)
- [`ReachabilityResult`](@ref)
"""
abstract type Result end

status(result::Result) = result.status

function validate_status(st::Symbol)
    @assert st ∈ (:holds, :violated, :unknown) "unexpected status code: `:$st`.\nOnly (:holds, :violated, :unknown) are accepted"
    return st
end


"""
    BasicResult(status::Symbol)

Result type that captures whether the input-output constraint is satisfied.
Possible status values:\n
    :holds (io constraint is satisfied always)\n
    :violated (io constraint is violated)\n
    :unknown (could not be determined)
"""
struct BasicResult <: Result
    status::Symbol
end


"""
ResultInfo(status, info)

Like `BasicResult`, but also returns a `info` dictionary that contains other informations.
This is designed to be the general result type.
"""
struct ResultInfo <: Result
    status::Symbol
    info::Dict
end

"""
    CounterExampleResult(status, counter_example)

Like `BasicResult`, but also returns a `counter_example` if one is found (if status = :violated).
The `counter_example` is a point in the input set that, after the NN, lies outside the output set.
"""
struct CounterExampleResult <: Result
    status::Symbol
    counter_example
    CounterExampleResult(s, ce) = new(validate_status(s), ce)
end

"""
    AdversarialResult(status, max_disturbance)

Like `BasicResult`, but also returns the maximum allowable disturbance in the input (if status = :violated).
"""
struct AdversarialResult <: Result
	status::Symbol
	max_disturbance::Float64
    AdversarialResult(s, md) = new(validate_status(s), md)
end

"""
    ReachabilityResult(status, reachable)

Like `BasicResult`, but also returns the output reachable set given the input constraint (if status = :violated).
"""
struct ReachabilityResult <: Result
	status::Symbol
	reachable
    ReachabilityResult(s, r) = new(validate_status(s), r)
end

# Additional constructors:
CounterExampleResult(s) = CounterExampleResult(s, Float64[])
AdversarialResult(s)    = AdversarialResult(s, -1.0)
ReachabilityResult(s)   = ReachabilityResult(s, AbstractPolytope[])

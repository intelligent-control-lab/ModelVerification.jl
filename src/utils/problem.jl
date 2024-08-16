"""
    get_chain(vertex)

Returns a `Flux.Chain` constructed from the given vertex. This is a helper 
function for `build_flux_model`. 

## Arguments
- `vertex`: Vertex from the `NaiveNASflux` computation graph.

## Returns
- `model`: `Flux.Chain` constructed from the given vertex.
- `curr_vertex`: The last vertex in the chain.
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

"""
    remove_flux_start_flatten(model::Chain)

Removes the starting `Flatten` layer from the model.

## Arguments
- `model` (`Chain`): `Flux.Chain` model.

## Returns
- `model` (`Chain`): `Flux.Chain` model with the starting `Flatten` layer 
    removed.
"""
function remove_flux_start_flatten(model::Chain)
    !isa(model[1], ONNXNaiveNASflux.Flatten) && return model
    println("removing starting flatten")
    return model[2:end]
end

"""
    purify_flux_model(model::Chain)

Removes the starting `Flatten` layer from the model. This is a wrapper function 
for `remove_flux_start_flatten`.

## Arguments
- `model` (`Chain`): `Flux.Chain` model.

## Returns
- `model` (`Chain`): `Flux.Chain` model with the starting `Flatten` layer 
    removed.
"""
function purify_flux_model(model::Chain)
    model = remove_flux_start_flatten(model)
end
 
"""
    get_shape(input::ImageConvexHull)

Returns the shape of the given `ImageConvexHull` input set.

## Arguments
- `input` (`ImageConvexHull`): Input set.

## Returns
- `shape` (`Tuple`): Shape of the input set. The last dimension is always the 
    number of the images. The first dimensions are the shape of the image. For 
    example, if the input set is consisted of 10 images of size 128 x 128, then 
    the shape is `(128, 128, 10)`.
"""
get_shape(input::ImageConvexHull) = (size(input.imgs[1])..., length(input.imgs))

"""
    get_shape(input::Hyperrectangle)

Returns the shape of the given `Hyperrectangle` input set.

## Arguments
- `input` (`Hyperrectangle`): Input set.

## Returns
- `shape` (`Tuple`): Shape of the hyperrectangle. The last dimension is always 
    1. For example, if the input set is a 2D hyperrectangle, then the shape is 
    `(2, 1)`.
"""
get_shape(input::Hyperrectangle) = (size(LazySets.center(input))..., 1)

"""
    build_onnx_model(path, model::Chain, input::InputSpec)

Builds an ONNX model from the given `Flux.Chain` model and input specification. 
The ONNX model is saved to the given path.

## Arguments
- `path` (`String`): Path to save the ONNX model.
- `model` (`Chain`): `Flux.Chain` model.
- `input` (`InputSpec`): Input specification.

## Returns
- `path` (`String`): Path to the saved ONNX model.
"""
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
- `input` : Input specification defined using a LazySet.
- `output` : Output specification defined using a LazySet.
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
ODEProblem(model::Chain, input, output) = Problem("", model, input, output)

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

"""
    status(result::Result)

Returns the status of the result. Only (:holds, :violated, :unknown) are 
accepted.
"""
status(result::Result) = result.status

"""
    validate_status(st)

Validates the status code. Only (:holds, :violated, :unknown) are accepted.

## Arguments
- `st` (`Symbol`): Status code.

## Returns
- Assertion Error if the given `st` is not one of (:holds, :violated, :unknown).
- Otherwise, returns the given `st`.
"""
function validate_status(st::Symbol)
    @assert st ∈ (:holds, :violated, :unknown) "unexpected status code: `:$st`.\nOnly (:holds, :violated, :unknown) are accepted"
    return st
end


"""
    BasicResult(status)

Result type that captures whether the input-output constraint is satisfied.
Possible status values:\n
    :holds (io constraint is satisfied always)\n
    :violated (io constraint is violated)\n
    :unknown (could not be determined)

## Fields
- `status` (`Symbol`): Status of the result, can be `:holds`, `:violated`, or 
    `:unknown`.
"""
struct BasicResult <: Result
    status::Symbol
end


"""
    ResultInfo(status, info)

Like `BasicResult`, but also returns a `info` dictionary that contains other 
informations. This is designed to be the general result type. 

## Fields
- `status` (`Symbol`): Status of the result, can be `:holds`, `:violated`, or 
    `:unknown`.
- `info` (`Dict`): A dictionary that contains information related to the result, 
    such as the verified bounds, adversarial input bounds, counter example, etc.
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
	max_disturbance::FloatType[]
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
CounterExampleResult(s) = CounterExampleResult(s, FloatType[][])
AdversarialResult(s)    = AdversarialResult(s, -1.0)
ReachabilityResult(s)   = ReachabilityResult(s, AbstractPolytope[])

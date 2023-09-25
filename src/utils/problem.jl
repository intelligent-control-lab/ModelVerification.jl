function get_parallel_chains(comp_vertices, index_more_than_one_outputs)
    function get_chain(vertex)
        m = Any[]
        curr_vertex = vertex
        while length(NaiveNASflux.inputs(curr_vertex)) == 1
            # println("curr vertex ", name(curr_vertex))
            push!(m, NaiveNASflux.layer(curr_vertex))
            curr_vertex = NaiveNASflux.outputs(curr_vertex)[1]
        end
        return Chain(m...), curr_vertex
    end
    outs = NaiveNASflux.outputs(comp_vertices[index_more_than_one_outputs])
    @assert length(outs) == 2
    chain1, vertex_more_than_one_inputs = get_chain(outs[1])
    chain2, _ = get_chain(outs[2])
    inner_iter = findfirst(v -> NaiveNASflux.name(v) == NaiveNASflux.name(vertex_more_than_one_inputs), comp_vertices)
    if length(chain1) == 0
        return SkipConnection(chain2, (+)), inner_iter
    elseif length(chain2) == 0
        return SkipConnection(chain1, (+)), inner_iter
    else
        return Parallel(+; α = chain1, β = chain2), inner_iter
    end
end

function build_flux_model(onnx_model_path)
    comp_graph = ONNXNaiveNASflux.load(onnx_model_path)
    model_vec = Any[]
    inner_iter = 0
    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        if length(string(NaiveNASflux.name(vertex))) >= 4 && string(NaiveNASflux.name(vertex))[1:4] == "data"
            continue
        end 
        push!(model_vec, NaiveNASflux.layer(vertex))
        if length(NaiveNASflux.outputs(vertex)) > 1
            parallel_chain, inner_iter = get_parallel_chains(ONNXNaiveNASflux.vertices(comp_graph), index)
            push!(model_vec, parallel_chain)
        end
    end
    model = Chain(model_vec...)
    Flux.testmode!(model)
    return (model)
end

get_shape(input::ImageConvexHull) = (size(input.imgs[1])..., length(input.imgs))
function build_onnx_model(path, model::Chain, input::InputSpec)
    ONNXNaiveNASflux.save(path, model, get_shape(input))
    return path
end

"""
    Problem{P, Q}(network::Network, input::P, output::Q)

Problem definition for neural verification.

The verification problem consists of: for all  points in the input set,
the corresponding output of the network must belong to the output set.
"""
struct Problem{P, Q}
    onnx_model_path::String
    Flux_model::Chain
    input::P
    output::Q
end
Problem(path::String, input_data, output_data) = #If the Problem only have onnx model input
    Problem(path, build_flux_model(path), input_data, output_data)
Problem(model::Chain, input_data, output_data) = #If the Problem only have Flux_mdoel input
    Problem(build_onnx_model("tmp.onnx", model, input_data), model, input_data, output_data) 

"""
    Result
Supertype of all result types.

See also: [`BasicResult`](@ref), [`CounterExampleResult`](@ref), [`AdversarialResult`](@ref), [`ReachabilityResult`](@ref)
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

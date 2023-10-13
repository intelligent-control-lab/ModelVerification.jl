struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope, Star}} <: SequentialForwardProp end
Ai2() = Ai2{Zonotope}()
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Ai2s = Ai2{Star}
const Box = Ai2{Hyperrectangle}  


struct StarSet <: SequentialForwardProp
    pre_bound_method::Union{SequentialForwardProp, Nothing}
end
StarSet() = StarSet(nothing)

function prepare_method(prop_method::StarSet, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, batch_output, model_info)
    if hasproperty(prop_method, :pre_bound_method) && !isnothing(prop_method.pre_bound_method)
        # batch_input = init_batch_bound(prop_method.pre_bound_method, batch_input, batch_output)
        pre_batch_info = init_propagation(prop_method.pre_bound_method, batch_input, batch_output, model_info)
        pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, batch_output, model_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            batch_info[node][:pre_bound] = pre_batch_info[prev_node][:bound]
        end
    end
    return batch_output, batch_info
end

function compute_bound(bound::Zonotope)
    radius = dropdims(sum(abs.(LazySets.genmat(bound)), dims=2), dims=2)
    return LazySets.center(bound) - radius, LazySets.center(bound) + radius
end

function compute_bound(bound::Star)
    box = overapproximate(bound, Hyperrectangle)
    return low(box), high(box)
end


function init_bound(prop_method::StarSet, input::Hyperrectangle) 
    isa(input, Star) && return input
    cen = LazySets.center(input) 
    gen = LazySets.genmat(input)
    T = eltype(input)
    n = dim(input)
    I = Matrix{T}(LinearAlgebra.I(n))
    A = [I; .-I]
    b = [ones(T, n); ones(T, n)] # -1 to 1
    return Star(T.(cen), T.(gen), HPolyhedron(A, b))  
end

function check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::LazySet)
    x = LazySets.center(input)
    # println(reach)
    # println(⊆(reach, output))
    ⊆(reach, output) && return ReachabilityResult(:holds, [reach])
    ∈(model(x), output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end

function check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::Complement)
    x = LazySets.center(input)
    unsafe_output = Complement(output)
    box_reach = box_approximation(reach)
    isdisjoint(box_reach, unsafe_output) && return ReachabilityResult(:holds, [reach])
    ∈(model(x), unsafe_output) && return CounterExampleResult(:violated, x)
    return CounterExampleResult(:unknown)
end

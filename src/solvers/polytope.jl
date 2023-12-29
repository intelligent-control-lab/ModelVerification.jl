"""
    Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope, Star}} 
                                    <: SequentialForwardProp

`Ai2` performs over-approximated reachability analysis to compute the over-
approximated output reachable set for a network. `T` can be `Hyperrectangle`, 
`Zonotope`, `Star`, or `HPolytope`. Different geometric representations impact
the verification performance due to different over-approximation sizes. 
We use `Zonotope` as "benchmark" geometry, as in the original implementation[1], 
due to improved scalability and precision (similar results can be achieved using 
`Star`). On the other hand, using a `HPolytope` representation potentially 
leads to a more precise but less scalable result, and the opposite holds for 
`Hyperrectangle`.

Note that initializing `Ai2()` defaults to `Ai2{Zonotope}`.
The following aliases also exist for convenience:

## Problem Requirement
1. Network: any depth, ReLU activation (more activations to be supported in the 
    future)
2. Input: AbstractPolytope
```Julia
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Ai2s = Ai2{Star}
const Box = Ai2{Hyperrectangle}
```
3. Output:  AbstractPolytope

## Returns
`ReachabilityResult`, `CounterExampleResult`

## Method
Reachability analysis using split and join.

## Property
Sound but not complete.

## Reference
[1] T. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and 
M. Vechev, "Ai2: Safety and Robustness Certification of Neural Networks with 
Abstract Interpretation," in *2018 IEEE Symposium on Security and Privacy (SP)*, 
2018.
"""
struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope, Star}} <: SequentialForwardProp end

Ai2() = Ai2{Zonotope}()

const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Ai2s = Ai2{Star}
const Box = Ai2{Hyperrectangle}  

"""
    StarSet

Covers supported Ai2 variations: Ai2h, Ai2z, Ai2s, Box.

## Fields
- `pre_bound_method` (`Union{SequentialForwardProp, Nothing}`): The geometric 
    representation used to compute the over-approximation of the input bounds.
"""
struct StarSet <: SequentialForwardProp
    pre_bound_method::Union{SequentialForwardProp, Nothing}
end
StarSet() = StarSet(nothing)

"""
    prepare_method(prop_method::StarSet, batch_input::AbstractVector, 
    batch_output::AbstractVector, model_info)

Initialize the bound of the start node of the computational graph for the 
`StarSet` solvers.

## Arguments
- `prop_method` (`StarSet`) : Propagation method of type `StarSet`.
- `batch_input` (`AbstractVector`) : Batch of inputs.
- `batch_output` (`AbstractVector`) : Batch of outputs.
- `model_info`: Structure containing the information of the neural network to
    be verified.

## Returns
- `batch_output`: batch of outputs.
- `batch_info`: Dictionary containing information of each node in the model.
"""
function prepare_method(prop_method::StarSet, batch_input::AbstractVector, 
                        batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, batch_output, 
                    model_info)
    if (hasproperty(prop_method, :pre_bound_method) 
        && !isnothing(prop_method.pre_bound_method))
        # batch_input = init_batch_bound(prop_method.pre_bound_method, 
        #                     batch_input, batch_output)
        pre_batch_info = init_propagation(prop_method.pre_bound_method, 
                            batch_input, batch_output, model_info)
        pre_batch_out_spec, pre_batch_info = 
            prepare_method(prop_method.pre_bound_method, batch_input, 
                batch_output, batch_inheritance, model_info)
        pre_batch_bound, pre_batch_info = 
            propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            batch_info[node][:pre_bound] = pre_batch_info[prev_node][:bound]
        end
    end
    return batch_output, batch_info
end

"""
    compute_bound(bound::Zonotope)

Computes the lower- and upper-bounds of a zonotope. 
This function is used when propagating through the layers of the model.
Radius is the sum of the absolute value of the generators of the given zonotope.

## Arguments
- `bound` (`Zonotope`) : zonotope of which the bounds need to be computed

## Returns
- Lower- and upper-bounds of the Zonotope.
"""
function compute_bound(bound::Zonotope)
    radius = dropdims(sum(abs.(LazySets.genmat(bound)), dims=2), dims=2)
    return LazySets.center(bound) - radius, LazySets.center(bound) + radius
end

"""
    compute_bound(bound::Star)

Computes the lower- and upper-bounds of a star set. 
This function is used when propagating through the layers of the model.
It overapproximates the given star set with a hyperrectangle.

## Arguments
- `bound` (`Star`): Star set of which the bounds need to be computed.

## Returns
- Lower- and upper-bounds of the overapproximated hyperrectangle.
"""
function compute_bound(bound::Star)
    box = overapproximate(bound, Hyperrectangle)
    return low(box), high(box)
end

"""
    init_bound(prop_method::StarSet, input::Hyperrectangle)

Given a hyperrectangle as `input`, this function returns a star set that 
encompasses the hyperrectangle. This helps a more precise computation of bounds.

## Arguments
- `prop_method` (`StarSet`): `StarSet`-type solver; includes `Ai2h`, `Ai2z`, 
    `Ai2s`, `Box`.
- `input` (`Hyperrectangle`): Hyperrectangle to be converted into a star set

## Returns
- `Star` set that encompasses the given hyperrectangle.
"""
function init_bound(prop_method::StarSet, input::Hyperrectangle) 
    isa(input, Star) && return input
    cen = LazySets.center(input) 
    gen = LazySets.genmat(input)
    T = eltype(input)
    n = dim(input)
    I = Matrix{T}(LinearAlgebra.I(n))
    A = [I; .-I]
    b = [ones(T, n); ones(T, n)] # -1 to 1
    s = Star(T.(cen), T.(gen), HPolyhedron(A, b))  
    # println(s)
    return s
end

"""
    check_inclusion(prop_method::ForwardProp, model, input::LazySet, 
    reach::LazySet, output::LazySet)

Determines whether the reachable set, `reach`, is within the valid 
output specified by a `LazySet`. This function achieves this by directly 
checking if the reachable set `reach` is a subset of the set of valid outputs 
`output`. If not, it attempts to find a counterexample and returns the 
appropriate `Result`.

## Arguments
- `prop_method` (`ForwardProp`): Solver being used.
- `model`: Neural network model that is to be verified.
- `input` (`LazySet`): Input specification supported by `LazySet`.
- `reach` (`LazySet`): Reachable set resulting from the propagation of `input` 
    through the `model`.
- `output` (`LazySet`) : Set of valid outputs represented with a `LazySet`.

## Returns
- `ReachabilityResult(:holds, [reach])` if `reach` is a subset of `output`.
- `CounterExampleResult(:unknown)` if `reach` is not a subset of `output`, but 
    cannot find a counterexample.
- `CounterExampleResult(:violated, x)` if `reach` is not a subset of `output`, 
    and there is a counterexample.
"""
function check_inclusion(prop_method::ForwardProp, model, input::LazySet, 
                         reach::LazySet, output::LazySet)
    # println(reach)
    # println(⊆(reach, output))
    ⊆(reach, output) && return ReachabilityResult(:holds, [reach])
    x = LazySets.center(input)
    ∈(model(x), output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
    # to = get_timer("Shared")
    # @timeit to "attack" return attack(model, input, output; restart=1)
end

"""
    check_inclusion(prop_method::ForwardProp, model, input::LazySet, 
    reach::LazySet, output::Complement)

Determines whether the reachable set, R(input, model), is within the valid 
output specified by a `LazySet`. This function achieves this by checking if the
box approximation (overapproximation with hyperrectangle) of the `reach` set is
disjoint with the `unsafe_output`. If the box approximation is a subset of the 
`unsafe_output`, then the safety property is violated. 

## Arguments
- `prop_method` (`ForwardProp`): Solver being used.
- `model`: Neural network model that is to be verified.
- `input` (`LazySet`): Input specification supported by `Lazyset`.
- `reach` (`LazySet`): Reachable set resulting from the propagation of `input` 
    through the `model`.
- `output` (`Complement`): Set of valid outputs represented with a complement 
    set. For problems using this `check_inclusion` method, the unsafe region is 
    specified. Then, the complement of the unsafe region is given as the desired 
    output specification.

## Returns
- `ReachabilityResult(:holds, [reach])` if `box_reach` is disjoint with the
    complement of the `output`.
- `CounterExampleResult(:violated, x)` if the center of the `input` set results 
    in a state that belongs to the `unsafe_output`.
- `CounterExampleResult(:unknown)` if either the two cases above are true.
"""
function check_inclusion(prop_method::ForwardProp, model, input::LazySet, 
                         reach::LazySet, output::Complement)
    x = LazySets.center(input)
    unsafe_output = Complement(output)
    box_reach = box_approximation(reach)
    isdisjoint(box_reach, unsafe_output) && return ReachabilityResult(:holds, [reach])
    ∈(model(x), unsafe_output) && return CounterExampleResult(:violated, x)
    return CounterExampleResult(:unknown)
    # to = get_timer("Shared")
    # @timeit to "attack" return attack(model, input, output; restart=1)
end

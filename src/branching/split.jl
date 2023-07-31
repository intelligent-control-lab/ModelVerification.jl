@with_kw struct Bisect <: SplitMethod
    num_split::Int64     = 1
end

function split_branch(split_method::Bisect, model::Chain, input::Hyperrectangle, output)
    split_method.num_split <= 0 && return [(input, output)]
    center, radius = LazySets.center(input), LazySets.radius_hyperrectangle(input)
    max_radius, max_idx = findmax(radius)
    input1, input2 = split_interval(input, max_idx)
    subtree1 = split_branch(Bisect(split_method.num_split-1), model, input1, output)
    subtree2 = split_branch(Bisect(split_method.num_split-1), model, input2, output)
    return [subtree1; subtree2]
end

function split_branch(split_method::Bisect, model::Chain, input::LazySet, output)
    return split_branch(split_method, model, box_approximation(input), output)
end


function split_branch(split_method::Bisect, model::Chain, input::ImageStarBound, output)
    println("splitting")
    @assert length(input.b) % 2 == 0
    n = length(input.b) ÷ 2
    T = eltype(input.A)
    I = Matrix{T}(LinearAlgebra.I(n))
    @assert all(input.A .≈ [I; .-I])
    u, l = input.b[1:n], .-input.b[n+1:end]
    max_radius, max_idx = findmax(u - l)
    bound1, bound2 = ImageStarBound(input.center, input.generators, input.A, input.b), ImageStarBound(input.center, input.generators, input.A, input.b)
    bound1.b[max_idx] = l[max_idx] + max_radius/2 # set new upper bound
    bound2.b[max_idx + n] = -(l[max_idx] + max_radius/2) # set new lower bound
    return [(bound1, output), (bound2, output)]
end


function split_branch(split_method::Bisect, model::Chain, input::ImageZonoBound, output)
    return [input, nothing] #TODO: find a way to split ImageZonoBound
end

"""
    split_interval(dom, i)

Split a set into two at the given index.

Inputs:
- `dom::Hyperrectangle`: the set to be split
- `i`: the index to split at
Return:
- `(left, right)::Tuple{Hyperrectangle, Hyperrectangle}`: two sets after split
"""

function split_interval(dom::Hyperrectangle, i::Int64)
    input_lower, input_upper = low(dom), high(dom)

    input_upper[i] = dom.center[i]
    input_split_left = Hyperrectangle(low = input_lower, high = input_upper)

    input_lower[i] = dom.center[i]
    input_upper[i] = dom.center[i] + dom.radius[i]
    input_split_right = Hyperrectangle(low = input_lower, high = input_upper)
    return (input_split_left, input_split_right)
end
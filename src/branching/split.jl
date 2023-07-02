@with_kw struct Bisect <: SplitMethod
    num_split::Int64     = 1
end

function split_branch(split_method::Bisect, model::Chain, input::Hyperrectangle, output, info)
    split_method.num_split <= 0 && return [(input, output, info)]
    center, radius = LazySets.center(input), LazySets.radius_hyperrectangle(input)
    max_radius, max_idx = findmax(radius)
    input1, input2 = split_interval(input, max_idx)
    subtree1 = split_branch(Bisect(split_method.num_split-1), model, input1, output, info)
    subtree2 = split_branch(Bisect(split_method.num_split-1), model, input2, output, info)
    return [subtree1; subtree2]
end

function split_branch(split_method::Bisect, model::Chain, input::LazySet, output, info)
    return split_branch(split_method, model, box_approximation(input), output, info)
end


function split_branch(split_method::Bisect, model::Chain, input::ImageStarBound, output, info)
    input.A
    
end


function split_branch(split_method::Bisect, model::Chain, input::ImageZonoBound, output, info)
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
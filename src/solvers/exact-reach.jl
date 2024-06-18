"""
    ExactReach <: SequentialForwardProp

`ExactReach` performs exact reachability analysis to compute the exact reachable 
set for a network. It works for piecewise linear networks with either linear or 
ReLU activations. It computes the reachable set for every linear segment of the 
network and keeps track of all sets. The final reachable set is the union of all 
sets. Since the number of linear segments is exponential in the number of nodes 
in one layer, this method is not scalable.

## Problem Requirement
1. Network: any depth, ReLU activation (more activations to be supported in the future)
2. Input: Array of AbstractPolytope, i.e., union of polytopes.
3. Output: Array of AbstractPolytope, i.e., union of polytopes.

## Returns
`BasicResult(:holds)`, `BasicResult(:violated)`

## Method
Reachability analysis using split and join.

## Property
Sound and complete.

## Reference
[1] C. Liu, T. Arnon, C. Lazarus, C. Strong, C. Barret, and M. J. Kochenderfer, 
"Algorithms for Verifying Deep Neural Networks," in _Foundations and Trends in 
Optimization_, 2021.

[2] W. Xiang, H.-D. Tran, and T. T. Johnson, "Reachable Set Computation and 
Safety Verification for Neural Networks with ReLU Activations," ArXiv Preprint 
_ArXiv:1712.08163_, 2017.
"""
struct ExactReach <: SequentialForwardProp end

"""
    ExactReachBound <: Bound

Bound for `ExactReach` solver. It is a union of polytopes, represented with an 
array of `LazySet`.

## Fields
- `polys` (`AbstractArray{LazySet}`): Array of polytopes.
"""
struct ExactReachBound <: Bound
    polys::AbstractArray{LazySet}
end

"""
get_center(bound::ExactReachBound)

Returns a randomly sampled point from the first polytope in the array of 
polytopes, `bound.polys`.

## Arguments
- `bound` (`ExactReachBound`): The `ExactReachBound` to sample from.

## Returns
- A randomly sampled point from the first polytope in the array of polytopes.
"""
function get_center(bound::ExactReachBound)
    return LazySets.sample(bound.polys[1])
end

"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, 
                    prop_method::ExactReach, problem::Problem)

Preprocessing of the `Problem` to be solved. This method converts the model to a 
bounded computational graph, makes the input specification compatible with the 
solver, and returns the model information and preprocessed `Problem`. This in 
turn also initializes the branch bank.

## Arguments
- `search_method` (`SearchMethod`): Method to search the branches.
- `split_method` (`SplitMethod`): Method to split the branches.
- `prop_method` (`ExactReach`): Solver to be used, specifically the 
    `ExactReach`.
- `problem` (`Problem`): Problem to be preprocessed to better fit the solver.

## Returns
- `model_info`, a structure containing the information of the neural network to 
    be verified.
- `Problem` after processing the initial input specification and model.
"""
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ExactReach, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
end

"""
    init_bound(prop_method::ExactReach, bound::LazySet)

For the `ExactReach` solver, this function converts the input set, represented 
with a `LazySet`, to an `ExactReachBound` representation. This serves as a 
preprocessing step for the `ExactReach` solver.

## Arguments
- `prop_method` (`ExactReach`): `ExactReach` solver.
- `bound` (`LazySet`): Input set, represented with a `LazySet`.

## Returns
- `ExactReachBound` representation of the input set.
"""
function init_bound(prop_method::ExactReach, bound::LazySet)
    return ExactReachBound([bound])
end

"""
    check_inclusion(prop_method::ExactReach, model, input::ExactReachBound, 
                    reach::ExactReachBound, output::LazySet)

Determines whether the reachable set, `reach`, is within the valid output 
specified by a `LazySet`. This function achieves this by directly checking if 
all the reachable sets in `reach` are subsets of the set of valid outputs 
`output`. If not, it returns `BasicResult(:violated)`. Otherwise, it returns 
`BasicResult(:holds)`.

## Arguments
- `prop_method` (`ExactReach`): Solver being used.
- `model`: Neural network model that is to be verified.
- `input` (`ExactReachBound`): Input specification represented with an 
    `ExactReachBound`.
- `reach` (`ExactReachBound`): Reachable set resulting from the propagation of
    `input` through the `model`, represented with an `ExactReachBound`.
- `output` (`LazySet`): Set of valid outputs represented with a `LazySet`.

## Returns
- `BasicResult(:holds)` if all reachable sets in `reach` are subsets of 
    `output`.
- `BasicResult(:violated)` if any reachable set in `reach` is not a subset of 
    `output`.
"""
function check_inclusion(prop_method::ExactReach, model, input::ExactReachBound, reach::ExactReachBound, output::LazySet)
    for bound in reach.polys
        ⊆(bound, output) && continue
        return BasicResult(:violated)
    end
    return BasicResult(:holds)
end

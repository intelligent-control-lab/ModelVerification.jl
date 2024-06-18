"""
    IBP <: BatchForwardProp

IBP is a verification approach that uses Box as the geometric 
representation. 
"""
struct IBP <: BatchForwardProp end

"""
    IBPBound{T<:Real} <: Bound

`IBPBound` is used to represent the bounded set for `IBP`.

## Fields
- `center` (`AbstractArray{T, 4}`): center image ("anchor" image in literature), 
    of size `heigth x width x number of channels x 1`.
- `generators` (`AbstractArray{T, 4}`): matrix of generator images, of size
    `height x width x number of channels x number of generators`.
"""
struct IBPBound <: Bound
    batch_low::AbstractArray    # reach_dim... x batch_size
    batch_up::AbstractArray     # reach_dim... x batch_size
end

"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, 
                    prop_method::IBP, problem::Problem)

Converts the model to a bounded computational graph and makes input 
specification compatible with the solver, `prop_method`. This in turn also 
initializes the branch bank.

## Arguments
- `search_method` (`SearchMethod`): Method to search the branches.
- `split_method` (`SplitMethod`): Method to split the branches.
- `prop_method` (`IBP`): Solver to be used, specifically the `IBP`.
- `problem` (`Problem`): Problem to be preprocessed to better fit the solver.

## Returns
- `model_info`, a structure containing the information of the neural network to 
    be verified.
- `Problem` after processing the initial input specification and model.
"""
# function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::IBP, problem::Problem)
#     model_info = onnx_parse(problem.onnx_model_path)
#     return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
# end

"""
    init_bound(prop_method::IBP, ch::ImageConvexHull) 

For the `IBP` solver, this function converts the input set, represented 
with an `ImageConvexHull`, to an `IBPBound` representation. This serves as 
a preprocessing step for the `IBP` solver. 

## Arguments
- `prop_method` (`IBP`): `IBP` solver.
- `ch` (`ImageConvexHull`): Convex hull, type `ImageConvexHull`, is used as the 
    input specification.

## Returns
- `IBPBound` set that encompasses the given `ImageConvexHull`.
"""

# function init_batch_bound(prop_method::IBP, batch_bouonds::AbstractArray, batch_output::AbstractArray) 
#     batch_low = cat([b.batch_low for b in batch_bouonds]..., dims=4)
#     batch_up = cat([b.batch_up for b in batch_bouonds]..., dims=4)
#     @assert all(batch_up - batch_low .>= 0)
#     @show size(batch_low)
#     return IBPBound(batch_low, batch_up)
# end

function init_batch_bound(prop_method::IBP, batch_chs::Vector{<:ImageConvexHull}, batch_output) 
    batch_low = cat([minimum(cat(ch.imgs..., dims=4), dims=(4)) for ch in batch_chs]..., dims=4)
    batch_up = cat([maximum(cat(ch.imgs..., dims=4), dims=(4)) for ch in batch_chs]..., dims=4)
    @assert all(batch_up - batch_low .>= 0)
    # @show size(batch_low)
    return IBPBound(batch_low, batch_up)
end
function init_batch_bound(prop_method::IBP, batch_box::Vector{<:Hyperrectangle}, batch_output) 
    batch_low = cat([low(box) for box in batch_box]..., dims=2)
    batch_up = cat([high(box) for box in batch_box]..., dims=2)
    @assert all(batch_up - batch_low .>= 0)
    # @show size(batch_low)
    return IBPBound(batch_low, batch_up)
end
"""
    compute_bound(bound::IBPBound)

Computes the lower- and upper-bounds of the box set.
This function is used when propagating through the layers of the model.
`compute_bound(bound::IBPBound
)`.

## Arguments
- `bound` (`IBPBound`): Image box set of which the bounds need to be 
    computed.

## Returns
- Lower- and upper-bounds.
"""
function compute_bound(bound::IBPBound)
    return bound.batch_low, bound.batch_up
end

"""
get_center(bound::IBPBound)

Returns the center image of the `IBPBound` bound.

## Arguments
- `bound` (`IBPBound`): Geometric representation of the specification 
    using `IBPBound`.

## Returns
- `IBPBound.center` image of type `AbstractArray{T, 4}`.
"""
get_center(bound::IBPBound) = bound.center[:,:,:,1]

"""
    check_inclusion(prop_method::IBP, model, input::IBPBound, 
                    reach::LazySet, output::LazySet)

Determines whether the reachable set, `reach`, is within the valid output 
specified by a `LazySet`.

## Agruments
- `prop_method` (`IBP`): Solver being used.
- `model`: Neural network model that is to be verified.
- `input` (`IBPBound`): Input specification supported by `IBPBound`.
- `reach` (`LazySet`): Reachable set resulting from the propagation of `input` 
    through the `model`.
- `output` (`LazySet`) : Set of valid outputs represented with a `LazySet`.

## Returns
- `ReachabilityResult(:holds, box_reach)` if `reach` is a subset of `output`, 
    the function returns `:holds` with the box approximation (overapproximation 
    with hyperrectangle) of the `reach` set.
- `CounterExampleResult(:unknown)` if `reach` is not a subset of `output`, but 
    cannot find a counterexample.
- `CounterExampleResult(:violated, x)` if `reach` is not a subset of `output`, 
    and there is a counterexample.
"""
function check_inclusion(prop_method::IBP, model, batch_chs::AbstractArray, bound::IBPBound, batch_outspec::AbstractArray)
    # box_reach = box_approximation(reach)
    ch = batch_chs[1]
    # @show size(mean(cat(ch.imgs..., dims=4), dims=(4)))
    # @show size(mean(ch.imgs, dims=(1,2,3)))
    centers = [mean(cat(ch.imgs..., dims=4), dims=(4)) for ch in batch_chs]
    batch_out = [FloatType[].(model(cen)) for cen in centers]
    batch_size = length(batch_chs)
    results = Result[BasicResult(:unknown) for _ in 1:batch_size]
    for i in 1:batch_size
        h = Hyperrectangle(low=bound.batch_low[:,i], high=bound.batch_up[:,i])
        ⊆(h, batch_outspec[i]) && (results[i] = BasicResult(:holds))
        ∉(batch_out[i][:,1], batch_outspec[i]) && (results[i] = CounterExampleResult(:violated, centers[i]))
        # @show low(h)
        # @show high(h)
    end
    return results
end

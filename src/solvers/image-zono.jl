"""
    ImageZono <: SequentialForwardProp

ImageZono is a verification approach that uses Image Zonotope as the geometric 
representation. It is an extension of `ImageStar` where there is no linear 
constraints on the free parameters, α:

``Θ = \\{ x : x = c + ∑_{i=1}^{m} (α_i v_i) \\}``

where ``c`` is the center image, ``V = \\{ v_1, …, v_m \\}`` is the set of
generator images, and α's are the free parameters.
"""
struct ImageZono <: SequentialForwardProp 
    use_gpu
end
ImageZono() = ImageZono(false)

"""
    ImageZonoBound{T<:Real} <: Bound

`ImageZonoBound` is used to represent the bounded set for `ImageZono`.

## Fields
- `center` (`AbstractArray{T, 4}`): center image ("anchor" image in literature), 
    of size `heigth x width x number of channels x 1`.
- `generators` (`AbstractArray{T, 4}`): matrix of generator images, of size
    `height x width x number of channels x number of generators`.
"""
struct ImageZonoBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  h x w x c x n_gen
end

"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, 
                    prop_method::ImageZono, problem::Problem)

Converts the model to a bounded computational graph and makes input 
specification compatible with the solver, `prop_method`. This in turn also 
initializes the branch bank.

## Arguments
- `search_method` (`SearchMethod`): Method to search the branches.
- `split_method` (`SplitMethod`): Method to split the branches.
- `prop_method` (`ImageZono`): Solver to be used, specifically the `ImageZono`.
- `problem` (`Problem`): Problem to be preprocessed to better fit the solver.

## Returns
- `model_info`, a structure containing the information of the neural network to 
    be verified.
- `Problem` after processing the initial input specification and model.
"""
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageZono, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
end

"""
    init_bound(prop_method::ImageZono, ch::ImageConvexHull) 

For the `ImageZono` solver, this function converts the input set, represented 
with an `ImageConvexHull`, to an `ImageZonoBound` representation. This serves as 
a preprocessing step for the `ImageZono` solver. 

## Arguments
- `prop_method` (`ImageZono`): `ImageZono` solver.
- `ch` (`ImageConvexHull`): Convex hull, type `ImageConvexHull`, is used as the 
    input specification.

## Returns
- `ImageZonoBound` set that encompasses the given `ImageConvexHull`.
"""
function init_bound(prop_method::ImageZono, ch::ImageConvexHull) 
    imgs = ch.imgs
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    return ImageZonoBound(cen, gen)
end

"""
    init_bound(prop_method::ImageZono, bound::ImageStarBound)

For the `ImageZono` solver, if the input set, represented with an 
`ImageStarBound`, is a zonotope, this function converts it to an 
`ImageZonoBound` representation.

## Arguments
- `prop_method` (`ImageZono`): `ImageZono` solver.
- `ch` (`ImageStarBound`): `ImageStarBound` is used for the input specification.

## Returns
- `ImageZonoBound` representation.
"""
function init_bound(prop_method::ImageZono, bound::ImageStarBound)
    assert_zono_star(bound)
    return ImageZonoBound(bound.center, bound.generators)
end

"""
    compute_bound(bound::ImageZonoBound)

Computes the lower- and upper-bounds of an image zono set.
This function is used when propagating through the layers of the model.
It converts the image zono set to a zonotope. Then, it computes the bounds using 
`compute_bound(bound::Zonotope)`.

## Arguments
- `bound` (`ImageZonoBound`): Image zono set of which the bounds need to be 
    computed.

## Returns
- Lower- and upper-bounds of the flattened zonotope.
"""
function compute_bound(bound::ImageZonoBound)
    cen = reshape(bound.center, :)
    gen = reshape(bound.generators, :, size(bound.generators,4))
    flat_reach = Zonotope(cen, gen)
    l, u = compute_bound(flat_reach)
    l = reshape(l, size(bound.center))
    u = reshape(u, size(bound.center))
    return l, u
end

"""
get_center(bound::ImageZonoBound)

Returns the center image of the `ImageZonoBound` bound.

## Arguments
- `bound` (`ImageZonoBound`): Geometric representation of the specification 
    using `ImageZonoBound`.

## Returns
- `ImageZonoBound.center` image of type `AbstractArray{T, 4}`.
"""
get_center(bound::ImageZonoBound) = bound.center[:,:,:,1]

"""
    check_inclusion(prop_method::ImageZono, model, input::ImageZonoBound, 
                    reach::LazySet, output::LazySet)

Determines whether the reachable set, `reach`, is within the valid output 
specified by a `LazySet`.

## Agruments
- `prop_method` (`ImageZono`): Solver being used.
- `model`: Neural network model that is to be verified.
- `input` (`ImageZonoBound`): Input specification supported by `ImageZonoBound`.
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
function check_inclusion(prop_method::ImageZono, model, input::ImageZonoBound, reach::LazySet, output::LazySet)
    # box_reach = box_approximation(reach)
    x = input.center
    y = reshape(model(x),:) # TODO: seems ad-hoc, the original last dimension is batch_size
    ⊆(reach, output) && return ReachabilityResult(:holds, reach)
    # ⊆(reach, output) && return ReachabilityResult(:holds, box_reach)
    ∈(y, output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end

function _isdisjoint(h::Hyperrectangle, p::HPolytope)
    A, b = tosimplehrep(p)
    # println(low(h))
    # println(high(h))
    lb, ub = interval_map(A, low(h), high(h))
    # println(lb)
    # println(ub)
    return any(ub .< b)
end

function check_inclusion(prop_method::ImageZono, model, input::ImageZonoBound, reach::LazySet, output::Complement)
    
    unsafe_output = Complement(output)
    box_reach = box_approximation(reach)
    _isdisjoint(box_reach, unsafe_output) && return ReachabilityResult(:holds, [reach])
    # isdisjoint(reach, unsafe_output) && return ReachabilityResult(:holds, [reach])
    
    sgn = [rand([-1, 1]) for _ in 1:size(input.generators,4)]
    rand_gen = sum(input.generators .* reshape(sgn, 1, 1, 1, :), dims=4)
    x = input.center .+ rand_gen
    y = model(x)
    ∈(vec(y), unsafe_output) && return CounterExampleResult(:violated, x)
    return CounterExampleResult(:unknown)
    # to = get_timer("Shared")
    # @timeit to "attack" return attack(model, input, output; restart=1)
end

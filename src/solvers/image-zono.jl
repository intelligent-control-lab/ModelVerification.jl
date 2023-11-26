"""
    ImageZono <: SequentialForwardProp

ImageZono is a verification approach that uses Image Zonotope as the geometric 
representation. It is an extension of `ImageStar` where there is no linear 
constraints on the free parameters, α:

``Θ = \\{ x : x = c + ∑_{i=1}^{m} (α_i v_i) \\}``

where ``c`` is the center image, ``V = \\{ v_1, …, v_m \\}`` is the set of
generator images, and α's are the free parameters.
"""
struct ImageZono <: SequentialForwardProp end

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
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageZono, problem::Problem)

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

For the `ImageStar` solver, this function converts the input set, represented 
with an `ImageConvexHull`, to an `ImageStarBound` representation. This serves as 
a preprocessing step for the `ImageStar` solver. It assumes that batch_input[1] 
is a list of vertex images. 
"""
function init_bound(prop_method::ImageZono, ch::ImageConvexHull) 
    imgs = ch.imgs
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    return ImageZonoBound(cen, gen)
end

"""
    init_bound(prop_method::ImageZono, bound::ImageStarBound)
"""
function init_bound(prop_method::ImageZono, bound::ImageStarBound)
    assert_zono_star(bound)
    return ImageZonoBound(bound.center, bound.generators)
end

"""
    compute_bound(bound::ImageZonoBound)
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

center(bound::ImageZonoBound) = bound.center

"""
    check_inclusion(prop_method::ImageZono, model, input::ImageZonoBound, reach::LazySet, output::LazySet)
"""
function check_inclusion(prop_method::ImageZono, model, input::ImageZonoBound, reach::LazySet, output::LazySet)
    # println(low(reach))
    # println(high(reach))
    box_reach = box_approximation(reach)
    x = input.center
    y = reshape(model(x),:) # TODO: seems ad-hoc, the original last dimension is batch_size
    ⊆(reach, output) && return ReachabilityResult(:holds, box_reach)
    ∈(y, output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end

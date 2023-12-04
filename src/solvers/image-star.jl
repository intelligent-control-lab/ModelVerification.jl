"""
    ImageStar <: SequentialForwardProp

ImageStar is a verification approach that can verify the robustness of 
Convolutional Neural Network (CNN). This toolbox uses the term, `ImageStar`, as 
the verification method itself that uses the ImageStar set. In terms of 
geometric representation, an ImageStar is an extension of the generalized star
set such that the center and generators are images with multiple channels.

``Θ = \\{ x : x = c + ∑_{i=1}^{m} (α_i v_i), \\; Cα ≤ d \\}``

where ``c`` is the center image, ``V = \\{ v_1, …, v_m \\}`` is the set of
generator images, and ``Cα ≤ d`` represent the predicate with α's as the free 
parameters. This set representation enables efficient over-approximative 
analysis of CNNs. ImageStar is less conservative and faster than ImageZono [1].

Note that initializing `ImageStar()` defaults to `ImageStar(nothing)`.

## Fields
- `pre_bound_method` (`Union{SequentialForwardProp, Nothing}`): The geometric 
    representation used to compute the over-approximation of the input bounds.

## Reference
[1] HD. Tran, S. Bak, W. Xiang, and T.T. Johnson, "Verification of Deep Convolutional 
Neural Networks Using ImageStars," in _Computer Aided Verification (CAV)_, 2020.
"""
struct ImageStar <: SequentialForwardProp 
    pre_bound_method::Union{SequentialForwardProp, Nothing}
end
ImageStar() = ImageStar(nothing)

"""
    ImageStarBound{T<:Real} <: Bound

`ImageStarBound` is used to represent the bounded set for `ImageStar`. 
It is an extension of the geometric representation, `StarSet`.

## Fields
- `center` (`AbstractArray{T, 4}`): center image ("anchor" image in literature), 
    of size `heigth x width x number of channels x 1`.
- `generators` (`AbstractArray{T, 4}`): matrix of generator images, of size
    `height x width x number of channels x number of generators`.
- `A` (`AbstractArray{T, 2}`): normal direction of the predicate, of size 
    `number of constraints x number of generators`.
- `b` (`AbstractArray{T, 1}`): constraints of the predicate, of size 
    `number of constraints x number of generators`.
"""
struct ImageStarBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   # h x w x c x n_gen
    A::AbstractArray{T, 2}            # n_con x n_gen
    b::AbstractArray{T, 1}            # n_con 
end

"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, 
                    prop_method::ImageStar, problem::Problem)

Preprocessing of the `Problem` to be solved. This method converts the model to a 
bounded computational graph, makes the input specification compatible with the 
solver, and returns the model information and preprocessed `Problem`. This in 
turn also initializes the branch bank.

## Arguments
- `search_method` (`SearchMethod`): Method to search the branches.
- `split_method` (`SplitMethod`): Method to split the branches.
- `prop_method` (`ImageStar`): Solver to be used, specifically the `ImageStar`.
- `problem` (`Problem`): Problem to be preprocessed to better fit the solver.

## Returns
- `model_info`, a structure containing the information of the neural network to 
    be verified.
- `Problem` after processing the initial input specification and model.
"""
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageStar, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
end

"""
    prepare_method(prop_method::ImageStar, batch_input::AbstractVector,
                   batch_output::AbstractVector, model_info)

Initialize the bound of the start node of the computational graph based on the 
`pre_bound_method` specified in the given ImageStar solver.

## Agruments
- `prop_method` (`ImageStar`): ImageStar solver.
- `batch_input` (`AbstractVector`): Batch of inputs.
- `batch_output` (`AbstractVector`): Batch of outputs.
- `model_info`: Structure containing the information of the neural network to
    be verified.

## Returns
- `batch_output`: Batch of outputs.
- `batch_info`: Dictionary containing information of each node in the model.
"""
prepare_method(prop_method::ImageStar, batch_input::AbstractVector, batch_output::AbstractVector, model_info) = prepare_method(StarSet(prop_method.pre_bound_method), batch_input, batch_output, model_info)

"""
    init_bound(prop_method::ImageStar, ch::ImageConvexHull) 

For the `ImageStar` solver, this function converts the input set, represented 
with an `ImageConvexHull`, to an `ImageStarBound` representation. This serves as 
a preprocessing step for the `ImageStar` solver. 

## Arguments
- `prop_method` (`ImageStar`): `ImageStar` solver.
- `ch` (`ImageConvexHull`): Convex hull, type `ImageConvexHull`, is used for the 
    input specification.

## Returns
- `ImageStarBound` set that encompasses the given `ImageConvexHull`.
"""
function init_bound(prop_method::ImageStar, ch::ImageConvexHull) 
    imgs = ch.imgs
    T = typeof(imgs[1][1,1,1])
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    n = length(imgs)-1 # number of generators
    I = Matrix{T}(LinearAlgebra.I(n))
    A = [I; .-I]
    b = [ones(T, n); ones(T, n)] # -1 to 1
    
    return ImageStarBound(T.(cen), T.(gen), A, b)
end

"""
    assert_zono_star(bound::ImageStarBound)

Asserts whether the given `ImageStarBound` set is a Zonotope.
This is done by checking whether the free parameter belongs to a unit hypercube.
"""
function assert_zono_star(bound::ImageStarBound)
    @assert length(bound.b) % 2 == 0
    n = length(bound.b) ÷ 2
    T = eltype(bound.A)
    I = Matrix{T}(LinearAlgebra.I(n))
    @assert all(bound.A .≈ [I; .-I])
    @assert all(bound.b == [ones(T, n); ones(T, n)]) # -1 to 1
end

"""
    compute_bound(bound::ImageStarBound)

Computes the lower- and upper-bounds of an image star set.
This function is used when propagating through the layers of the model.
It converts the image star set to a star set. Then, it overapproximates this 
star set with a hyperrectangle.

## Arguments
- `bound` (`ImageStarBound`): Image star set of which the bounds need to be 
    computed.

## Returns
- Lower- and upper-bounds of the overapproximated hyperrectangle.
"""
function compute_bound(bound::ImageStarBound)
    cen = reshape(bound.center, :)
    gen = reshape(bound.generators, :, size(bound.generators,4))
    flat_reach = ImageStar_to_Star(bound)
    l, u = compute_bound(flat_reach)
    l = reshape(l, size(bound.center))
    u = reshape(u, size(bound.center))
    return l, u
end

"""
    center(bound::ImageStarBound)

Returns the center image of the `ImageStarBound` bound.

## Arguments
- `bound` (`ImageStarBound`): Geometric representation of the specification 
    using `ImageStarBound`.

## Returns
- `ImageStarBound.center` image of type `AbstractArray{T, 4}`.
"""
center(bound::ImageStarBound) = bound.center

"""
    check_inclusion(prop_method::ImageStar, model, input::ImageStarBound, 
                    reach::LazySet, output::LazySet)

Determines whether the reachable set, `reach`, is within the valid output 
specified by a `LazySet`. 

## Agruments
- `prop_method` (`ImageStar`): Solver being used.
- `model`: Neural network model that is to be verified.
- `input` (`ImageStarBound`): Input specification supported by `ImageStarBound`.
- `reach` (`LazySet`): Reachable set resulting from the propagation of `input` 
    through the `model`.
- `output` (`LazySet`): Set of valid outputs represented with a `LazySet`.

## Returns
- `ReachabilityResult(:holds, box_reach)` if `reach` is a subset of `output`, 
    the function returns `:holds` with the box approximation (overapproximation 
    with hyperrectangle) of the `reach` set.
- `CounterExampleResult(:unknown)` if `reach` is not a subset of `output`, but 
    cannot find a counterexample.
- `CounterExampleResult(:violated, x)` if `reach` is not a subset of `output`, 
    and there is a counterexample.
"""
function check_inclusion(prop_method::ImageStar, model, input::ImageStarBound, reach::LazySet, output::LazySet)
    box_reach = box_approximation(reach)
    # println(low(reach))
    # println(high(reach))
    x_coe = sample(HPolyhedron(input.A, input.b))
    x_coe = reshape(x_coe, 1, 1, 1, length(x_coe))
    vec = dropdims(sum(input.generators .* x_coe, dims=4), dims=4)
    x = input.center + vec # input.center may not be inside of the inputset
    # display(heatmap(reshape(x, (28,28))))
    # println(x)
    # println("reach")
    # println(reach)
    # println("output")
    # println(output)
    y = reshape(model(x),:) # TODO: seems ad-hoc, the original last dimension is batch_size
    ⊆(reach, output) && return ReachabilityResult(:holds, box_reach)
    ∈(y, output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end



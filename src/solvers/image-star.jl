struct ImageStar <: SequentialForwardProp 
    pre_bound_method::Union{SequentialForwardProp, Nothing}
end
ImageStar() = ImageStar(nothing)

struct ImageStarBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  h x w x c x n_gen
    A::AbstractArray{T, 2}            # n_con x n_gen
    b::AbstractArray{T, 1}            # n_con 
end

"""
init_bound(prop_method::ImageStar, batch_input) 

Assume batch_input[1] is a list of vertex images.
Return a zonotope. 

Outputs:
- `ImageStarBound`
"""


function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageStar, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
end


prepare_method(prop_method::ImageStar, batch_input::AbstractVector, batch_output::AbstractVector, model_info) = prepare_method(StarSet(prop_method.pre_bound_method), batch_input, batch_output, model_info)

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

function assert_zono_star(bound::ImageStarBound)
    @assert length(bound.b) % 2 == 0
    n = length(bound.b) ÷ 2
    T = eltype(bound.A)
    I = Matrix{T}(LinearAlgebra.I(n))
    @assert all(bound.A .≈ [I; .-I])
    @assert all(bound.b == [ones(T, n); ones(T, n)]) # -1 to 1
end

function compute_bound(bound::ImageStarBound)
    cen = reshape(bound.center, :)
    gen = reshape(bound.generators, :, size(bound.generators,4))
    flat_reach = ImageStar_to_Star(bound)
    l, u = compute_bound(flat_reach)
    l = reshape(l, size(bound.center))
    u = reshape(u, size(bound.center))
    return l, u
end

center(bound::ImageStarBound) = bound.center

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



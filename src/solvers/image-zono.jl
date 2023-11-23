
struct ImageZono <: SequentialForwardProp end

struct ImageZonoBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  h x w x c x n_gen
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageZono, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
end

function init_bound(prop_method::ImageZono, ch::ImageConvexHull) 
    imgs = ch.imgs
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    return ImageZonoBound(cen, gen)
end

function init_bound(prop_method::ImageZono, bound::ImageStarBound)
    assert_zono_star(bound)
    return ImageZonoBound(bound.center, bound.generators)
end

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

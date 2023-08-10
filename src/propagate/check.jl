
function check_inclusion(prop_method::ForwardProp, model, input, bound::ImageZonoBound, output)
    z = Zonotope(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)))
    box_reach = box_approximation(z)
    println("Image Zono reach")
    # println(z)
    # println("bound")
    # println(bound)
    println(volume(box_reach))
    return ReachabilityResult(:holds, box_reach)
end

function check_inclusion(prop_method::ForwardProp, model, input, bound::ImageStarBound, output)
    return ReachabilityResult(:holds, bound)
end

function check_inclusion(prop_method::ImageStar, model, input::Union{ImageZonoBound, ImageStarBound}, reach::LazySet, output::LazySet)
    box_reach = box_approximation(reach)
    println(volume(box_reach))
    ⊆(reach, output) && return ReachabilityResult(:holds, box_reach)
    x = input.center
    y = reshape(model(x),:) # TODO: seems ad-hoc, the original last dimension is batch_size
    ∈(y, output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end

function check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::LazySet)
    x = LazySets.center(input)
    # println(reach)
    # println(⊆(reach, output))
    ⊆(reach, output) && return ReachabilityResult(:holds, [reach])
    ∈(model(x), output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end

function check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::Complement)
    x = LazySets.center(input)
    unsafe_output = Complement(output)
    box_reach = box_approximation(reach)
    isdisjoint(box_reach, unsafe_output) && return ReachabilityResult(:holds, [reach])
    ∈(model(x), unsafe_output) && return CounterExampleResult(:violated, x)
    return CounterExampleResult(:unknown)
end

function check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, batch_reach::AbstractArray, batch_output::AbstractArray)
    results = [check_inclusion(prop_method, model, batch_input[i], batch_reach[i], batch_output[i]) for i in eachindex(batch_reach)]
    return results
end

function check_inclusion(prop_method::Crown, model, batch_input::AbstractArray, bound::CrownBound, batch_out_spec::LinearSpec)
    # l, u: out_dim x batch_size
    l, u = compute_bound(bound)
    batch_size = size(l,2)
    #pos_A = max.(batch_out_spec.A, fmap(cu, zeros(size(batch_out_spec.A)))) # spec_dim x out_dim x batch_size
    #neg_A = min.(batch_out_spec.A, fmap(cu, zeros(size(batch_out_spec.A))))
    pos_A = clamp.(batch_out_spec.A, 0, Inf)
    neg_A = clamp.(batch_out_spec.A, -Inf, 0)
    spec_u = batched_vec(pos_A, u) + batched_vec(neg_A, l) .- batch_out_spec.b # spec_dim x batch_size
    spec_l = batched_vec(pos_A, l) + batched_vec(neg_A, u) .- batch_out_spec.b # spec_dim x batch_size
    CUDA.@allowscalar center = (bound.batch_data_min[1:end-1,:] + bound.batch_data_max[1:end-1,:])./2 # out_dim x batch_size
    out_center = model(center)
    center_res = batched_vec(batch_out_spec.A, out_center) .- batch_out_spec.b # spec_dim x batch_size
    results = [BasicResult(:unknown) for _ in 1:batch_size]
    spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
    spec_l = reshape(maximum(spec_l, dims=1), batch_size) # batch_size, min_x max_i of ai x - bi
    println("spec")
    println(spec_l)
    println(spec_u)
    center_res = reshape(maximum(center_res, dims=1), batch_size) # batch_size
    if batch_out_spec.is_complement 
        # A x < b descript the unsafe set, violated if exist x such that max spec ai x - bi <= 0    
        for i in 1:batch_size
            CUDA.@allowscalar center_res[i] <= 0 && (results[i] = BasicResult(:violated))
            CUDA.@allowscalar spec_l[i] > 0 && (results[i] = BasicResult(:holds))
        end
    else # holds if forall x such that max spec ai x - bi <= 0
        for i in 1:batch_size
            CUDA.@allowscalar spec_u[i] <= 0 && (results[i] = BasicResult(:holds))
            CUDA.@allowscalar center_res[i] > 0 && (results[i] = BasicResult(:violated))
        end
    end
    
    return results
end

function check_inclusion(prop_method::AlphaCrown, model, batch_input::AbstractArray, bound::ConcretizeCrownBound, batch_out_spec::LinearSpec)
    # spec_l, spec_u = process_bound(prop_method::AlphaCrown, bound, batch_out_spec, batch_info)
    batch_input = prop_method.use_gpu ? fmap(cu, batch_input) : batch_input
    spec_l, spec_u = bound.spec_l, bound.spec_u
    batch_size = length(batch_input)
    #center = (bound.batch_data_min[1:end,:] + bound.batch_data_max[1:end,:])./2 # out_dim x batch_size
    center = (bound.batch_data_min .+ bound.batch_data_max) ./ 2 # out_dim x batch_size
    model = prop_method.use_gpu ? fmap(cu, model) : model
    out_center = model(center)
    center_res = batched_vec(batch_out_spec.A, out_center) .- batch_out_spec.b # spec_dim x batch_size
    center_res = reshape(maximum(center_res, dims=1), batch_size) # batch_size
    results = [BasicResult(:unknown) for _ in 1:batch_size]

    # complement out spec: violated if exist y such that Ay-b < 0. Need to make sure lower bound of Ay-b > 0 to hold, spec_l > 0
    if batch_out_spec.is_complement
        @assert prop_method.bound_lower 
        spec_l = reshape(maximum(spec_l, dims=1), batch_size) # batch_size, min_x max_i of ai x - bi
        if prop_method.use_gpu
            for i in 1:batch_size
                CUDA.@allowscalar center_res[i] <= 0 && (results[i] = BasicResult(:violated))
                CUDA.@allowscalar spec_l[i] > 0 && (results[i] = BasicResult(:holds))
            end
        else
            for i in 1:batch_size
                center_res[i] <= 0 && (results[i] = BasicResult(:violated))
                spec_l[i] > 0 && (results[i] = BasicResult(:holds))
            end
        end
    else # polytope out spec: holds if all y such that Ay-b < 0. Need to make sure upper bound of Ay-b < 0 to hold.
        @assert prop_method.bound_upper
        spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
        if prop_method.use_gpu
            for i in 1:batch_size
                CUDA.@allowscalar spec_u[i] <= 0 && (results[i] = BasicResult(:holds))
                CUDA.@allowscalar center_res[i] > 0 && (results[i] = BasicResult(:violated))
            end
        else
            for i in 1:batch_size
                CUDA.@allowscalar spec_u[i] <= 0 && (results[i] = BasicResult(:holds))
                CUDA.@allowscalar center_res[i] > 0 && (results[i] = BasicResult(:violated))
            end
        end
    end
    return results
end


function check_inclusion(prop_method::BetaCrown, model, batch_input::AbstractArray, bound::ConcretizeCrownBound, batch_out_spec::LinearSpec)
    # spec_l, spec_u = process_bound(prop_method::AlphaCrown, bound, batch_out_spec, batch_info)
    batch_input = prop_method.use_gpu ? fmap(cu, batch_input) : batch_input
    spec_l, spec_u = bound.spec_l, bound.spec_u
    batch_size = length(batch_input)
    #center = (bound.batch_data_min[1:end,:] + bound.batch_data_max[1:end,:])./2 # out_dim x batch_size
    center = (bound.batch_data_min .+ bound.batch_data_max) ./ 2 # out_dim x batch_size
    model = prop_method.use_gpu ? fmap(cu, model) : model
    out_center = model(center)
    center_res = batched_vec(batch_out_spec.A, out_center) .- batch_out_spec.b # spec_dim x batch_size
    center_res = reshape(maximum(center_res, dims=1), batch_size) # batch_size
    results = [BasicResult(:unknown) for _ in 1:batch_size]

    # complement out spec: violated if exist y such that Ay-b < 0. Need to make sure lower bound of Ay-b > 0 to hold, spec_l > 0
    if batch_out_spec.is_complement
        @assert prop_method.bound_lower 
        spec_l = reshape(maximum(spec_l, dims=1), batch_size) # batch_size, min_x max_i of ai x - bi
        for i in 1:batch_size
            CUDA.@allowscalar center_res[i] <= 0 && (results[i] = BasicResult(:violated))
            CUDA.@allowscalar spec_l[i] > 0 && (results[i] = BasicResult(:holds))
        end
    else # polytope out spec: holds if all y such that Ay-b < 0. Need to make sure upper bound of Ay-b < 0 to hold.
        @assert prop_method.bound_upper
        spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
        for i in 1:batch_size
            CUDA.@allowscalar spec_u[i] <= 0 && (results[i] = BasicResult(:holds))
            CUDA.@allowscalar center_res[i] > 0 && (results[i] = BasicResult(:violated))
        end
    end
    return results
end 
"""
    batched_interval_map(W::Matrix, l::AbstractVecOrMat, u::AbstractVecOrMat)

Simple linear mapping on intervals.
`L, U := ([W]₊*l + [W]₋*u), ([W]₊*u + [W]₋*l)`

Outputs:
- `(lbound, ubound)` (after the mapping)
"""
function batched_interval_map(W::AbstractMatrix, l::AbstractArray{T,2}, u::AbstractArray{T,2}) where T
    # W : A x B
    # l : B x batch

    l_new = max.(W, zeros(size(W))) * l + min.(W, zeros(size(W))) * u
    u_new = max.(W, zeros(size(W))) * u + min.(W, zeros(size(W))) * l
    return (l_new, u_new)
end

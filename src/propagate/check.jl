
function check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::LazySet)
    # println("--")
    #println("input ", low(input), high(input))
    #println("reach ", low(reach), high(reach))
    x = LazySets.center(input)
    #println("center ", x, " ", model(x))
    #println("result ", ⊆(reach, output) ? :holds : ∈(model(x), output) ? :unknown : :violated)
    ⊆(reach, output) && return ReachabilityResult(:holds, [reach])
    ∈(model(x), output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end

function check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::Complement)
    # println("checking inclusion")
    x = LazySets.center(input)
    unsafe_output = Complement(output)
    # println("checking empty")
    # println(unsafe_output)
    box_reach = box_approximation(reach)
    # println(box_reach)
    # println(isdisjoint(box_reach, unsafe_output))
    # println("===")
    # println("done isempty")
    # ∈(model(x), unsafe_output)
    # println("done issubset")
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
    l, u = concretize(bound)
    # println("l u")
    # println(l)
    # println(u)
    # println("A")
    # println(batch_out_spec.A)
    # println("b")
    # println(batch_out_spec.b)
    # @assert all(l.<=u) "check lower bound larger than upper bound"
    batch_size = size(l,2)
    pos_A = max.(batch_out_spec.A, zeros(size(batch_out_spec.A))) # spec_dim x out_dim x batch_size
    neg_A = min.(batch_out_spec.A, zeros(size(batch_out_spec.A)))
    spec_u = batched_vec(pos_A, u) + batched_vec(neg_A, l) .- batch_out_spec.b # spec_dim x batch_size
    spec_l = batched_vec(pos_A, l) + batched_vec(neg_A, u) .- batch_out_spec.b # spec_dim x batch_size
    center = (bound.batch_data_min[1:end-1,:] + bound.batch_data_max[1:end-1,:])./2 # out_dim x batch_size
    out_center = model(center)
    center_res = batched_vec(batch_out_spec.A, out_center) .- batch_out_spec.b # spec_dim x batch_size
    results = [BasicResult(:unknown) for _ in 1:batch_size]
    # println("spec_l")
    # println(spec_l)
    # println("spec_u")
    # println(spec_u)
    
    spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
    spec_l = reshape(maximum(spec_l, dims=1), batch_size) # batch_size, min_x max_i of ai x - bi
    center_res = reshape(maximum(center_res, dims=1), batch_size) # batch_size
    
    if batch_out_spec.is_complement 
        # A x < b descript the unsafe set, violated if exist x such that max spec ai x - bi <= 0    
        for i in 1:batch_size
            center_res[i] <= 0 && (results[i] = BasicResult(:violated))
            spec_l[i] > 0 && (results[i] = BasicResult(:holds))
        end
    else # holds if forall x such that max spec ai x - bi <= 0
        for i in 1:batch_size
            spec_u[i] <= 0 && (results[i] = BasicResult(:holds))
            center_res[i] > 0 && (results[i] = BasicResult(:violated))
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

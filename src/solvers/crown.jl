struct Crown <: BatchForwardProp 
    use_gpu
    bound_lower::Bool
    bound_upper::Bool
end


struct CrownBound <: Bound
    batch_Low    # reach_dim x input_dim+1 x batch_size
    batch_Up     # reach_dim x input_dim+1 x batch_size
    batch_data_min    # input_dim+1 x batch_size
    batch_data_max     # input_dim+1 x batch_size
end

struct ConcretizeCrownBound <: Bound
    spec_l
    spec_u
    batch_data_min
    batch_data_max
end



function prepare_method(prop_method::Crown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, batch_output, model_info)
    out_specs = get_linear_spec(batch_output)
    if prop_method.use_gpu
        out_specs = LinearSpec(fmap(cu, out_specs.A), fmap(cu, out_specs.b), fmap(cu, out_specs.is_complement))
    end
    return out_specs, batch_info
end


function init_batch_bound(prop_method::Crown, batch_input::AbstractArray, batch_output::AbstractArray)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    n = prop_method.use_gpu ? fmap(cu, dim(batch_input[1])) : dim(batch_input[1])
    I = prop_method.use_gpu ? fmap(cu, Matrix{Float64}(LinearAlgebra.I(n))) : Matrix{Float64}(LinearAlgebra.I(n))
    Z = prop_method.use_gpu ? fmap(cu, zeros(n)) : zeros(n)
    batch_Low = prop_method.use_gpu ? fmap(cu, repeat([I Z], outer=(1, 1, batch_size))) : repeat([I Z], outer=(1, 1, batch_size))
    batch_Up = prop_method.use_gpu ? fmap(cu, repeat([I Z], outer=(1, 1, batch_size))) : repeat([I Z], outer=(1, 1, batch_size))
    batch_data_min = prop_method.use_gpu ? fmap(cu, cat([low(h) for h in batch_input]..., dims=2)) : cat([low(h) for h in batch_input]..., dims=2)
    batch_data_min = prop_method.use_gpu ? fmap(cu, [batch_data_min; ones(batch_size)']) : [batch_data_min; ones(batch_size)']# the last dimension is for bias
    batch_data_max = prop_method.use_gpu ? fmap(cu, cat([high(h) for h in batch_input]..., dims=2)) : cat([high(h) for h in batch_input]..., dims=2)
    batch_data_max = prop_method.use_gpu ? fmap(cu, [batch_data_max; ones(batch_size)']) : [batch_data_max; ones(batch_size)']# the last dimension is for bias
    bound = CrownBound(batch_Low, batch_Up, batch_data_min, batch_data_max)
    return bound
end

"""   
compute_bound(low::AbstractVecOrMat, up::AbstractVecOrMat, data_min_batch, data_max_batch) where N

Compute lower and upper bounds of a relu node in Crown.
`l, u := ([low]₊*data_min + [low]₋*data_max), ([up]₊*data_max + [up]₋*data_min)`

Outputs:
- `(lbound, ubound)`
"""
function compute_bound(bound::CrownBound)
    # low::AbstractVecOrMat{N}, up::AbstractVecOrMat{N}, data_min_batch, data_max_batch
    # low : reach_dim x input_dim x batch
    # data_min_batch: input_dim x batch
    # l: reach_dim x batch
    # batched_vec is a mutant of batched_mul that accepts batched vector as input.
    #z =   fmap(cu, zeros(size(bound.batch_Low)))
    #l =   batched_vec(max.(bound.batch_Low, z), bound.batch_data_min) + batched_vec(min.(bound.batch_Low, z), bound.batch_data_max)
    #u =   batched_vec(max.(bound.batch_Up, z), bound.batch_data_max) + batched_vec(min.(bound.batch_Up, z), bound.batch_data_min)
    l =   batched_vec(clamp.(bound.batch_Low, 0, Inf), bound.batch_data_min) .+ batched_vec(clamp.(bound.batch_Low, -Inf, 0), bound.batch_data_max)
    u =   batched_vec(clamp.(bound.batch_Up, 0, Inf), bound.batch_data_max) .+ batched_vec(clamp.(bound.batch_Up, -Inf, 0), bound.batch_data_min)
    # @assert all(l.<=u) "lower bound larger than upper bound"
    return l, u
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
    # println("typeof(center)")
    # println(typeof(center))
    # println("typeof(model)")
    # println(typeof(model))
    model = model |> gpu
    out_center = model(center)
    # println("typeof(out_center)")
    # println(typeof(out_center))
    center_res = batched_vec(batch_out_spec.A, out_center) .- batch_out_spec.b # spec_dim x batch_size
    results = [BasicResult(:unknown) for _ in 1:batch_size]
    spec_u = reshape(maximum(spec_u, dims=1), batch_size) # batch_size, max_x max_i of ai x - bi
    spec_l = reshape(maximum(spec_l, dims=1), batch_size) # batch_size, min_x max_i of ai x - bi
    # println("spec")
    # println(spec_l)
    # println(spec_u)
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
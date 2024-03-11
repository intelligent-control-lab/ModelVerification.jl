@enum CrownBoundHeuristics zero_slope parallel_slope adaptive_slope

"""
    Crown <: BatchForwardProp 
"""
struct Crown <: BatchForwardProp 
    use_gpu
    bound_lower::Bool
    bound_upper::Bool
    bound_heuristics::CrownBoundHeuristics
end
Crown(;use_gpu=true, bound_lower=true, bound_upper=true, bound_heuristics=zero_slope) = Crown(use_gpu, bound_lower, bound_upper, bound_heuristics)

"""
    CrownBound <: Bound
"""
mutable struct CrownBound <: Bound
    batch_Low    # reach_dim x input_dim+1 x batch_size
    batch_Up     # reach_dim x input_dim+1 x batch_size
    batch_data_min    # input_dim+1 x batch_size
    batch_data_max     # input_dim+1 x batch_size
    img_size    # width x height x channel or nothing if the input is not ImageConvexHull
end
function CrownBound(batch_Low, batch_Up, batch_data_min, batch_data_max)
    return CrownBound(batch_Low, batch_Up, batch_data_min, batch_data_max, nothing)
end

"""
    ConcretizeCrownBound <: Bound
"""
struct ConcretizeCrownBound <: Bound
    spec_l
    spec_u
    batch_data_min
    batch_data_max
end

"""
    prepare_problem(search_method::SearchMethod, split_method::SplitMethod, 
                    prop_method::Crown, problem::Problem)
"""
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::Crown, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    model = prop_method.use_gpu ? fmap(cu, problem.Flux_model) : problem.Flux_model
    return model_info, Problem(problem.onnx_model_path, model, init_bound(prop_method, problem.input), problem.output)
end

prepare_method(prop_method::Crown, batch_input::AbstractVector, batch_output::AbstractVector, batch_inheritance::AbstractVector, model_info) =
    prepare_method(prop_method, batch_input, get_linear_spec(batch_output),[nothing], model_info)

"""
    prepare_method(prop_method::Crown, batch_input::AbstractVector, 
                   out_specs::LinearSpec, model_info)
"""    
function prepare_method(prop_method::Crown, batch_input::AbstractVector, out_specs::LinearSpec, batch_inheritance::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, out_specs, model_info)
    if prop_method.use_gpu
        out_specs = LinearSpec(fmap(cu, out_specs.A), fmap(cu, out_specs.b), fmap(cu, out_specs.is_complement))
    end
    return out_specs, batch_info
end

"""
    init_batch_bound(prop_method::Crown, batch_input::AbstractArray, out_specs)
"""
function init_batch_bound(prop_method::Crown, batch_input::AbstractArray, out_specs)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    img_size = nothing
    if typeof(batch_input[1]) == ReLUConstrainedDomain
        batch_input = [b.domain for b in batch_input]
    end
    if typeof(batch_input[1]) == ImageConvexHull
        # convert batch_input from list of ImageConvexHull to list of Hyperrectangle
        img_size = ModelVerification.get_size(batch_input[1])
        @assert all(<=(0), batch_input[1].imgs[1]-batch_input[1].imgs[2]) "the first ImageConvexHull input must be upper bounded by the second ImageConvexHull input"
        batch_input = [Hyperrectangle(low = reduce(vcat,img_CH.imgs[1]), high = reduce(vcat,img_CH.imgs[2]))  for img_CH in batch_input]
    end
    n = prop_method.use_gpu ? fmap(cu, dim(batch_input[1])) : dim(batch_input[1])
    I = prop_method.use_gpu ? fmap(cu, Matrix{Float64}(LinearAlgebra.I(n))) : Matrix{Float64}(LinearAlgebra.I(n))
    Z = prop_method.use_gpu ? fmap(cu, zeros(n)) : zeros(n)
    batch_Low = prop_method.use_gpu ? fmap(cu, repeat([I Z], outer=(1, 1, batch_size))) : repeat([I Z], outer=(1, 1, batch_size))
    batch_Up = prop_method.use_gpu ? fmap(cu, repeat([I Z], outer=(1, 1, batch_size))) : repeat([I Z], outer=(1, 1, batch_size))
    batch_data_min = prop_method.use_gpu ? fmap(cu, cat([low(h) for h in batch_input]..., dims=2)) : cat([low(h) for h in batch_input]..., dims=2)
    batch_data_min = prop_method.use_gpu ? fmap(cu, [batch_data_min; ones(batch_size)']) : [batch_data_min; ones(batch_size)']# the last dimension is for bias
    batch_data_max = prop_method.use_gpu ? fmap(cu, cat([high(h) for h in batch_input]..., dims=2)) : cat([high(h) for h in batch_input]..., dims=2)
    batch_data_max = prop_method.use_gpu ? fmap(cu, [batch_data_max; ones(batch_size)']) : [batch_data_max; ones(batch_size)']# the last dimension is for bias
    if !isnothing(img_size) 
        # restore the image size for lower and upper bounds
        batch_Low = reshape(batch_Low, (img_size..., size(batch_Low)[2],size(batch_Low)[3]))
        batch_Up = reshape(batch_Up, (img_size..., size(batch_Up)[2],size(batch_Up)[3]))
    end
    # @show size(batch_data_min)
    bound = CrownBound(batch_Low, batch_Up, batch_data_min, batch_data_max, img_size)
    # @show bound
    # @show size(reshape(batch_Low, (img_size..., size(batch_Low)[2],size(batch_Low)[3])))
    return bound
end



"""   
    compute_bound(bound::CrownBound)

Compute lower and upper bounds of a relu node in Crown.
`l, u := ([low]₊*data_min + [low]₋*data_max), ([up]₊*data_max + [up]₋*data_min)`

## Arguments
- `bound` (`CrownBound`): CrownBound object
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
    if length(size(bound.batch_Low)) > 3
        img_size = size(bound.batch_Low)[1:3]
        batch_Low= reshape(bound.batch_Low, (img_size[1]*img_size[2]*img_size[3], size(bound.batch_Low)[4],size(bound.batch_Low)[5]))
        batch_Up= reshape(bound.batch_Up, (img_size[1]*img_size[2]*img_size[3], size(bound.batch_Up)[4],size(bound.batch_Up)[5]))
        l =   batched_vec(clamp.(batch_Low, 0, Inf), bound.batch_data_min) .+ batched_vec(clamp.(batch_Low, -Inf, 0), bound.batch_data_max)
        u =   batched_vec(clamp.(batch_Up, 0, Inf), bound.batch_data_max) .+ batched_vec(clamp.(batch_Up, -Inf, 0), bound.batch_data_min)
    else
        l =   batched_vec(clamp.(bound.batch_Low, 0, Inf), bound.batch_data_min) .+ batched_vec(clamp.(bound.batch_Low, -Inf, 0), bound.batch_data_max)
        u =   batched_vec(clamp.(bound.batch_Up, 0, Inf), bound.batch_data_max) .+ batched_vec(clamp.(bound.batch_Up, -Inf, 0), bound.batch_data_min)
    end
    # @assert all(l.<=u) "lower bound larger than upper bound"
    return l, u
end

"""
    compute_bound(bound::ConcretizeCrownBound)
"""
function compute_bound(bound::ConcretizeCrownBound)
    return bound.spec_l, bound.spec_u
end

"""
    check_inclusion(prop_method::Crown, model, batch_input::AbstractArray, bound::CrownBound, batch_out_spec::LinearSpec)
"""
function check_inclusion(prop_method::Crown, model, batch_input::AbstractArray, bound::CrownBound, batch_out_spec::LinearSpec)
    # l, u: out_dim x batch_size
    l, u = compute_bound(bound)
    # @show size(l)
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
    model = prop_method.use_gpu ? model |> gpu : model
    # @show model[end-3:end]
    # @show size(center)
    if !isnothing(bound.img_size)
        # resize input to match Conv for image
        # @assert length(size(bound.img_size)) == 3
        center = reshape(center, (bound.img_size..., size(center)[2]))
    end
    out_center = model(center)

    # TODO: uncomment the following if using Box Conv
    # num_last_dense=0
    # for i = 1:length(model)
    #     if model[length(model)+1-i] isa Dense
    #         num_last_dense += 1
    #     else
    #         break
    #     end
    # end
    
    # if num_last_dense == length(model)
    #     dense_model = model
    # else
    #     dense_model = model[end-num_last_dense:end]
    # end
    # out_center = dense_model(center)

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
            CUDA.@allowscalar center_res[i] <= -tol && (results[i] = BasicResult(:violated))
            CUDA.@allowscalar spec_l[i] > -tol && (results[i] = BasicResult(:holds))
        end
    else # holds if forall x such that max spec ai x - bi <= tol
        for i in 1:batch_size
            CUDA.@allowscalar spec_u[i] <= tol && (results[i] = BasicResult(:holds))
            CUDA.@allowscalar center_res[i] > tol && (results[i] = BasicResult(:violated))
        end
    end
    
    return results
end

"""
    convert the flatten Crown bound into a image-resized Crown bound
"""
function convert_CROWN_Bound_batch(flatten_bound::CrownBound, img_size)
    @assert length(size(flatten_bound.batch_Low)) == 3
    output_Low, output_Up = copy(flatten_bound.batch_Low), copy(flatten_bound.batch_Up) 
    output_Low= reshape(output_Low, (img_size..., size(flatten_bound.batch_Low)[2],size(flatten_bound.batch_Low)[3]))
    output_Up= reshape(output_Up, (img_size..., size(flatten_bound.batch_Up)[2],size(flatten_bound.batch_Up)[3]))
    new_bound = CrownBound(output_Low, output_Up, flatten_bound.batch_data_min, flatten_bound.batch_data_max, flatten_bound.img_size)
    return new_bound
end

"""
    convert the image-resized  Crown bound into a flatten Crown bound
"""
function convert_CROWN_Bound_batch(img_bound::CrownBound)
    @assert length(size(img_bound.batch_Low)) > 3
    img_size = size(img_bound.batch_Low)[1:3]
    output_Low, output_Up = copy(img_bound.batch_Low), copy(img_bound.batch_Up) 
    output_Low= reshape(output_Low, (img_size[1]*img_size[2]*img_size[3], size(img_bound.batch_Low)[4],size(img_bound.batch_Low)[5]))
    output_Up= reshape(output_Up, (img_size[1]*img_size[2]*img_size[3], size(img_bound.batch_Up)[4],size(img_bound.batch_Up)[5]))
    new_bound = CrownBound(output_Low, output_Up, img_bound.batch_data_min, img_bound.batch_data_max, img_bound.img_size)
    return new_bound, img_size
end

function get_center(bound::CrownBound)
    @show size(bound.batch_Low)
    @show size(bound.batch_data_min)
    l, u = compute_bound(bound)
    @show size(l)
    @show size(u)
    if isnothing(bound.img_size)
        return (l+u)./2
    else
        img_center = reshape((l+u)./2, (bound.img_size..., size(l)[2]))
        return img_center
    end
end


abstract type Bound end

function init_batch_bound(prop_method::ForwardProp, batch_input, batch_output)
    return [init_bound(prop_method, input) for input in batch_input]
end

function init_batch_bound(prop_method::BackwardProp, batch_input, batch_output)
    return [init_bound(prop_method, output) for output in batch_output]
end

function init_bound(prop_method::ForwardProp, input)
    return input
end

function init_bound(prop_method::BackwardProp, output)
    return output
end


struct ImageStarBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  h x w x c x n_gen
    A::AbstractArray{T, 2}            # n_con x n_gen
    b::AbstractArray{T, 1}            # n_con 
end

struct ImageZonoBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  h x w x c x n_gen
end


"""
init_bound(prop_method::ImageStar, batch_input) 

Assume batch_input[1] is a list of vertex images.
Return a zonotope. 

Outputs:
- `ImageStarBound`
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

function init_bound(prop_method::ImageZono, ch::ImageConvexHull) 
    imgs = ch.imgs
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    return ImageZonoBound(cen, gen)
end

function assert_zono_star(bound::ImageStarBound)
    @assert length(bound.b) % 2 == 0
    n = length(bound.b) ÷ 2
    T = eltype(bound.A)
    I = Matrix{T}(LinearAlgebra.I(n))
    @assert all(bound.A .≈ [I; .-I])
    @assert all(bound.b == [ones(T, n); ones(T, n)]) # -1 to 1
end
function init_bound(prop_method::ImageZono, bound::ImageStarBound)
    assert_zono_star(bound)
    return ImageZonoBound(bound.center, bound.generators)
end


# function compute_bound(bound::Zonotope)
#     l, u = low(bound), high(bound)
#     return l, u
# end

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
center(bound::ImageStarBound) = bound.center

function init_bound(prop_method::StarSet, input::Hyperrectangle) 
    isa(input, Star) && return input
    cen = LazySets.center(input) 
    gen = LazySets.genmat(input)
    T = eltype(input)
    n = dim(input)
    I = Matrix{T}(LinearAlgebra.I(n))
    A = [I; .-I]
    b = [ones(T, n); ones(T, n)] # -1 to 1
    return Star(T.(cen), T.(gen), HPolyhedron(A, b))  
end


struct LinearBound{T<:Real, F<:AbstractPolytope} <: Bound
    Low::AbstractArray{T, 3} # reach_dim x input_dim x batch_size
    Up::AbstractArray{T, 3}  # reach_dim x input_dim x batch_size
    domain::AbstractArray{F}  
end

struct CrownBound{T<:Real} <: Bound
    batch_Low::AbstractArray{T, 3}    # reach_dim x input_dim+1 x batch_size
    batch_Up::AbstractArray{T, 3}     # reach_dim x input_dim+1 x batch_size
    batch_data_min::AbstractArray{T, 2}     # input_dim+1 x batch_size
    batch_data_max::AbstractArray{T, 2}     # input_dim+1 x batch_size
end

struct AlphaCrownBound <: Bound
    lower_A_x
    upper_A_x
    lower_A_W
    upper_A_W
    batch_data_min
    batch_data_max
end

struct ConcretizeCrownBound <: Bound
    spec_l
    spec_u
    batch_data_min
    batch_data_max
end

function init_batch_bound(prop_method::Crown, batch_input::AbstractArray, batch_output::AbstractArray)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    n = dim(batch_input[1])
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    batch_Low = repeat([I Z], outer=(1, 1, batch_size))
    batch_Up = repeat([I Z], outer=(1, 1, batch_size))
    batch_data_min = cat([low(h) for h in batch_input]..., dims=2)
    batch_data_min = [batch_data_min; ones(batch_size)'] # the last dimension is for bias
    batch_data_max = cat([high(h) for h in batch_input]..., dims=2)
    batch_data_max = [batch_data_max; ones(batch_size)'] # the last dimension is for bias
    bound = CrownBound(batch_Low, batch_Up, batch_data_min, batch_data_max)
    return bound
end

function init_batch_bound(prop_method::AlphaCrown, batch_input::AbstractArray, batch_output::LinearSpec)
    batch_data_min = cat([low(h) for h in batch_input]..., dims=2)
    batch_data_max = cat([high(h) for h in batch_input]..., dims=2)

    bound = AlphaCrownBound([], [], nothing, nothing, batch_data_min, batch_data_max)
    return bound

    # spec_layer(W, b) = x -> [NNlib.batched_mul(x[1], W), NNlib.batched_mul(x[1], b) .+ x[2]]
    
    # # complement out spec: violated if exist y such that Ay-b < 0. Need to make sure lower bound of Ay-b > 0 to hold
    # # polytope out spec: holds if all y such that Ay-b < 0. Need to make sure upper bound of Ay-b < 0 to hold.
    # lA_x = batch_output.is_complement ? Vector{Any}([spec_layer(batch_output.A, batch_output.b)]) : []
    # uA_x = batch_output.is_complement ? [] : Vector{Any}([spec_layer(batch_output.A, batch_output.b)])
    
    # bound = AlphaCrownBound(lA_x, uA_x, nothing, nothing, batch_data_min, batch_data_max)
    
    # return bound
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
    z = zeros(size(bound.batch_Low))
    l =   batched_vec(max.(bound.batch_Low, z), bound.batch_data_min) + batched_vec(min.(bound.batch_Low, z), bound.batch_data_max)
    u =   batched_vec(max.(bound.batch_Up, z), bound.batch_data_max) + batched_vec(min.(bound.batch_Up, z), bound.batch_data_min)
    # @assert all(l.<=u) "lower bound larger than upper bound"
    return l, u
end


struct Compute_bound
    batch_data_min
    batch_data_max
end
Flux.@functor Compute_bound ()

function (f::Compute_bound)(x)
    z = zeros(size(x[1]))
    l = batched_mul(max.(x[1], z), f.batch_data_min) .+ batched_mul(min.(x[1], z), f.batch_data_max) .+ x[2]
    u = batched_mul(max.(x[1], z), f.batch_data_max) .+ batched_mul(min.(x[1], z), f.batch_data_min) .+ x[2]
    return l, u
end 

function process_bound(prop_method::PropMethod, batch_bound, batch_out_spec, batch_info)
    return batch_bound, batch_info
end


function optimize_bound(model, input, loss_func, optimizer, max_iter)
    opt_state = Flux.setup(optimizer, model)
    for i in 1 : max_iter
        losses, grads = Flux.withgradient(model) do m
            result = m(input) 
            loss_func(result)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    return model
end


function init_A_b(n, batch_size)
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    A = repeat(I, outer=(1, 1, batch_size))
    b = repeat(Z, outer=(1, 1, batch_size))
    return [A, b]
end

function process_bound(prop_method::AlphaCrown, batch_bound, batch_out_spec, batch_info)
    #println("batch_bound.batch_data_min max")
    #println(batch_bound.batch_data_min, batch_bound.batch_data_max)
    compute_bound = Compute_bound(batch_bound.batch_data_min, batch_bound.batch_data_max)

    bound_model = Chain(push!(prop_method.bound_lower ? batch_bound.lower_A_x : batch_bound.upper_A_x, compute_bound))
    # maximize lower(A * x - b) or minimize upper(A * x - b)
    loss_func = prop_method.bound_lower ?  x -> - sum(x[1]) : x -> sum(x[2])
    
    bound_model = optimize_bound(bound_model, batch_info[:spec_A_b], loss_func, prop_method.optimizer, prop_method.train_iteration)
    
    for (index, params) in enumerate(Flux.params(bound_model))
        relu_node = batch_info[:Alpha_Lower_Layer_node][index]
        batch_info[relu_node][prop_method.bound_lower ? :alpha_lower : :alpha_upper] = params
    end

    #println("batch_info[:spec_A_b]")
    #println(batch_info[:spec_A_b])

    spec_l, spec_u = bound_model(batch_info[:spec_A_b])
    
    # spec_A: spec_dim x out_dim x batch_dim
    # spec_l: spec_dim x batch_dim

    # println("prop_method.bound_lower")
    # println(prop_method.bound_lower)
    # println("prop_method.bound_upper")
    # println(prop_method.bound_upper)
    # n = size(batch_out_spec.A, 2)
    # batch_size = size(batch_out_spec.A, 3)
    # out_l, out_u = bound_model(init_A_b(n, batch_size)) # out_dim x batch_dim
    # println("out_l, out_u")
    # println(out_l, out_u)
    #println("spec_l, spec_u")
    #println(spec_l, spec_u)
    #println("spec_A, spec_b")
    #println(batch_out_spec.A, batch_out_spec.b)
    return ConcretizeCrownBound(spec_l, spec_u, batch_bound.batch_data_min, batch_bound.batch_data_max), batch_info
end


function set_used_nodes(final_node, batch_info, Model) #finish verifying
    push!(Model, "last_final_node" => nothing)
    if final_node != Model["last_final_node"]
        push!(Model, "last_final_node" => final_node)
        push!(Model, "used_nodes" => [])
        for node in Model["all_nodes"]
            push!(batch_info[node], "used" => false)
        end
        push!(batch_info[final_node], "used" => true)
        queue = Queue{Any}()
        enqueue!(queue, final_node)
        while !isempty(queue)
            node = dequeue!(queue)
            push!(Model["used_nodes"], node)
            if isnothing(batch_info[node]["inputs"])
                continue
            else
                for input_node in batch_info[node]["inputs"]
                    if !batch_info[input_node]["used"]
                        push!(batch_info[input_node], "used" => true)
                        enqueue!(queue, input_node)
                    end
                end
            end
        end
    end
end


struct GradientBound{F<:AbstractPolytope, N<:Real}
    sym::LinearBound{F} # reach_dim x input_dim x batch_size
    LΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
    UΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
end

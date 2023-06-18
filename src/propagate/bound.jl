
abstract type Bound end


function init_bound(prop_method::PropMethod, batch_input)
    return batch_input
end

struct ImageStarBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  of  h x w x 4
    P::HPolyhedron                          # n_con x n_gen+1
end

struct ImageZonoBound{T<:Real} <: Bound
    center::AbstractArray{T, 4}       # h x w x c x 1
    generators::AbstractArray{T, 4}   #  of  h x w x 4
end

"""
init_bound(prop_method::ImageStar, batch_input) 

Assume batch_input[1] is a list of vertex images.
Return a zonotope. 

Outputs:
- `ImageStarBound`
"""
function init_bound(prop_method::ImageStar, batch_input) 
    # batch_input = [list of vertex images]
    @assert length(batch_input) == 1 "ImageStarBound only support batch_size = 1"
    imgs = batch_input[1]
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    n = length(imgs)-1 # number of generators
    T = typeof(imgs[1][1,1,1])
    I = Matrix{T}(LinearAlgebra.I(n))
    P = HPolyhedron([I; .-I], [ones(T, n); ones(T, n)]) # -1 to 1
    return ImageStarBound(cen, gen, P)
end

function init_bound(prop_method::ImageStarZono, batch_input) 
    # batch_input = [list of vertex images]
    @assert length(batch_input) == 1 "ImageStarBound only support batch_size = 1"
    imgs = batch_input[1]
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    return ImageZonoBound(cen, gen)
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

function init_bound(prop_method::Crown, batch_input::AbstractArray)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    n = dim(batch_input[1])
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    batch_Low = repeat([I Z], outer=(1, 1, batch_size))
    batch_Up = repeat([I Z], outer=(1,1, batch_size))
    batch_data_min = cat([low(h) for h in batch_input]..., dims=2)
    # println("init bound")
    # println(size(batch_data_min))
    # println(size(zeros(batch_size)))
    batch_data_min = [batch_data_min; ones(batch_size)'] # the last dimension is for bias
    batch_data_max = cat([high(h) for h in batch_input]..., dims=2)
    batch_data_max = [batch_data_max; ones(batch_size)'] # the last dimension is for bias
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
    z = zeros(size(bound.batch_Low))
    # println(size(bound.batch_Low))
    # println(size(bound.batch_data_min))
    # println("compute_bound")
    # println("bound.batch_Low")
    # println(bound.batch_Low)
    # println("bound.batch_Up")
    # println(bound.batch_Up)
    # println("bound.batch_data_min")
    # println(bound.batch_data_min)
    # println("bound.batch_data_max")
    # println(bound.batch_data_max)
    
    l =   batched_vec(max.(bound.batch_Low, z), bound.batch_data_min) + batched_vec(min.(bound.batch_Low, z), bound.batch_data_max)
    u =   batched_vec(max.(bound.batch_Up, z), bound.batch_data_max) + batched_vec(min.(bound.batch_Up, z), bound.batch_data_min)
    # println("compute_bound")
    # println("l")
    # println(l)
    # println("u")
    # println(u)
    # println(l.<=u)
    # @assert all(l.<=u) "lower bound larger than upper bound"
    return l, u
end

struct GradientBound{F<:AbstractPolytope, N<:Real}
    sym::LinearBound{F} # reach_dim x input_dim x batch_size
    LΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
    UΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
end

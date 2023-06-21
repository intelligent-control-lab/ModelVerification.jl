
abstract type Bound end

function init_batch_bound(prop_method::PropMethod, batch_input)
    return [init_bound(prop_method, input) for input in batch_input]
end

function init_bound(prop_method::ForwardProp, input)
    return input
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

function init_bound(prop_method::ImageStar, imgs::AbstractVector) 
    T = typeof(imgs[1][1,1,1])
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    n = length(imgs)-1 # number of generators
    I = Matrix{T}(LinearAlgebra.I(n))
    A = [I; .-I]
    b = [ones(T, n); ones(T, n)] # -1 to 1
    return ImageStarBound(T.(cen), T.(gen), A, b)
end

function init_bound(prop_method::ImageStarZono, imgs::AbstractVector) 
    cen = cat([imgs[1] .+ sum([0.5 .* (img .- imgs[1]) for img in imgs[2:end]])]..., dims=4)
    gen = cat([0.5 .* (img .- imgs[1]) for img in imgs[2:end]]..., dims=4)
    return ImageZonoBound(cen, gen)
end

function init_bound(prop_method::Ai2s, input::Hyperrectangle) 
    isa(input, Star) && return input
    cen = center(input) 
    gen = genmat(input)
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

function init_batch_bound(prop_method::Crown, batch_input::AbstractArray)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    n = dim(batch_input[1])
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    batch_Low = repeat([I Z], outer=(1, 1, batch_size))
    batch_Up = repeat([I Z], outer=(1,1, batch_size))
    batch_data_min = cat([low(h) for h in batch_input]..., dims=2)
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
    l =   batched_vec(max.(bound.batch_Low, z), bound.batch_data_min) + batched_vec(min.(bound.batch_Low, z), bound.batch_data_max)
    u =   batched_vec(max.(bound.batch_Up, z), bound.batch_data_max) + batched_vec(min.(bound.batch_Up, z), bound.batch_data_min)
    # @assert all(l.<=u) "lower bound larger than upper bound"
    return l, u
end

struct GradientBound{F<:AbstractPolytope, N<:Real}
    sym::LinearBound{F} # reach_dim x input_dim x batch_size
    LΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
    UΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
end

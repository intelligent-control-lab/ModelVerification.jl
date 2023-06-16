
abstract type Bound end


function init_bound(prop_method::PropMethod, batch_input)
    return batch_input
end

struct LinearBound{F<:AbstractPolytope} <: Bound
    Low::AbstractArray{Float64, 3} # reach_dim x input_dim x batch_size
    Up::AbstractArray{Float64, 3}  # reach_dim x input_dim x batch_size
    domain::AbstractArray{F}  
end

struct CrownBound <: Bound
    batch_Low::AbstractArray{Float64, 3}    # reach_dim x input_dim+1 x batch_size
    batch_Up::AbstractArray{Float64, 3}     # reach_dim x input_dim+1 x batch_size
    batch_data_min::AbstractArray{Float64, 2}     # input_dim+1 x batch_size
    batch_data_max::AbstractArray{Float64, 2}     # input_dim+1 x batch_size
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

struct GradientBound{F<:AbstractPolytope, N<:Real}
    sym::LinearBound{F} # reach_dim x input_dim x batch_size
    LΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
    UΛ::Vector{Vector{N}}    # reach_dim x input_dim x batch_size
end

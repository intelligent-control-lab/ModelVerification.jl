# batch_lower_reach(reach::LinearBound) = AffineMap(@view(reach.Low[:, 1:end-1,:]), reach.domain, @view(reach.Low[:, end ,:]))

# upper_bound(a::AbstractVector, set::LazySet) = a'σ(a, set)
# lower_bound(a::AbstractVector, set::LazySet) = a'σ(-a, set) # ≡ -ρ(-a, set)
# bounds(a::AbstractVector, set::LazySet) = (a'σ(-a, set), a'σ(a, set))  # (lower, upper)

# upper_bound(S::LazySet, j::Integer) = upper_bound(Arrays.SingleEntryVector(j, dim(S), 1.0), S)
# lower_bound(S::LazySet, j::Integer) = lower_bound(Arrays.SingleEntryVector(j, dim(S), 1.0), S)
# bounds(S::LazySet, j::Integer) = (lower_bound(S, j), upper_bound(S, j))


"""
    broadcast_mid_dim(m::AbstractArray{2}, target::AbstractArray{T,3})

Given a target tensor of the shape AxBxC, 
broadcast the 2D mask of the shape AxC to AxBxC.

Outputs:
- `m` broadcasted.
"""
function broadcast_mid_dim(m::AbstractArray{T1,2}, target::AbstractArray{T2,3}) where T1 where T2
    @assert size(m,1) == size(target,1) "Size mismatch in broadcast_mid_dim"
    @assert size(m,2) == size(target,3) "Size mismatch in broadcast_mid_dim"
    # reshape the mask to match the shape of target
    m = reshape(m, size(m, 1), 1, size(m, 2)) # reach_dim x 1 x batch
    # replicate the mask along the second dimension
    m = repeat(m, 1, size(target, 2), 1)
    return m
end

function forward_act(prop_method::Crown, layer::typeof(relu), bound::CrownBound, batch_info)
    
    output_Low, output_Up = copy(bound.batch_Low), copy(bound.batch_Up) # reach_dim x input_dim x batch

    # If the lower bound of the lower bound is positive,
    # No change to the linear bounds.
    
    # If the upper bound of the upper bound is negative, set
    # both linear bounds to 0
    l, u = concretize(bound) # reach_dim x batch
    
    inact_mask = u .<= 0 # reach_dim x batch
    inact_mask_ext = broadcast_mid_dim(inact_mask, output_Low) # reach_dim x input_dim x batch
    output_Low[inact_mask_ext] .= 0
    output_Up[inact_mask_ext] .= 0

    
    # if the bounds overlap 0, concretize by setting
    # the generators to 0, and setting the new upper bound
    # center to be the current upper-upper bound.
    unstable_mask = (u .> 0) .& (l .< 0) # reach_dim x batch
    unstable_mask_ext = broadcast_mid_dim(unstable_mask, output_Low) # reach_dim x input_dim+1 x batch
    slope = u[unstable_mask] ./ (u[unstable_mask] .- l[unstable_mask]) # selected_reach_dim * selected_batch
    slope_mtx = ones(size(u))

    slope_mtx[unstable_mask] = u[unstable_mask] ./ (u[unstable_mask] .- l[unstable_mask]) # reach_dim x batch
    broad_slope = broadcast_mid_dim(slope_mtx, output_Up) # selected_reach_dim x input_dim+1 x selected_batch
    # broad_slop = reshape(slope, )
    output_Up .*= broad_slope
    unstable_mask_bias = copy(unstable_mask_ext)
    unstable_mask_bias[:,1:end-1,:] .= 0
    output_Up[unstable_mask_bias] .+= (slope .* max.(-u[unstable_mask], 0))[:]

    # output_Low[unstable_mask_ext] .*= broad_slope[:]
    output_Low[unstable_mask_ext] .= 0

    @assert !any(isnan, output_Low) "relu low contains NaN"
    @assert !any(isnan, output_Up) "relu up contains NaN"
    
    new_bound = CrownBound(output_Low, output_Up, bound.batch_data_min, bound.batch_data_max)
    l, u = concretize(new_bound)

    return new_bound, batch_info
end


# function forward_act(prop_method::ReluVal, layer::typeof(relu), batch_reach::LinearBound, batch_info)
    
#     output_Low, output_Up = copy(batch_reach.Low), copy(batch_reach.Up) # reach_dim x input_dim x batch
#     n_node = size(batch_reach,1)
#     LΛᵢ, UΛᵢ = zeros(n_node, n_batch), ones(n_node, n_batch)

#     # If the upper bound of the upper bound is negative, set
#     # the generators and centers of both bounds to 0, and
#     # the gradient mask to 0
#     inact_mask = upper_bound(batch_upper_reach(input)) .<= 0
#     LΛᵢ[inact_mask], UΛᵢ[inact_mask] = 0, 0
#     output_Low[inact_mask] .= 0
#     output_Up[inact_mask] .= 0

#     # If the lower bound of the lower bound is positive,
#     # the gradient mask should be 1
#     act_mask = lower_bound(batch_lower_reach(input)) .>= 0
#     LΛᵢ[act_mask], UΛᵢ[act_mask] = 1, 1

#     # if the bounds overlap 0, concretize by setting
#     # the generators to 0, and setting the new upper bound
#     # center to be the current upper-upper bound.
#     unstable_mask = ~(inact_mask | act_mask) 
#     LΛᵢ[unstable_mask] .= 0
#     UΛᵢ[unstable_mask] .= 1
#     output_Low[unstable_mask] .= 0
#     if lower_bound(batch_upper_reach(input), j) < 0
#         output_Up[j, :] .= 0
#         output_Up[j, end] = upper_bound(batch_upper_reach(input), j)
#     end

#     sym = SymbolicInterval(output_Low, output_Up, domain(input))
#     LΛ = push!(input.LΛ, LΛᵢ)
#     UΛ = push!(input.UΛ, UΛᵢ)
#     return SymbolicIntervalGradient(sym, LΛ, UΛ)
# end

# function forward_act(prop_method::Neurify, layer::typeof(relu), batch_reach::LinearBound, batch_info)
    
#     output_Low, output_Up = copy(batch_reach.Low), copy(batch_reach.Up) # reach_dim x input_dim x batch
#     n_node = n_nodes(L)
#     LΛᵢ, UΛᵢ = zeros(n_node), ones(n_node)

#     up_low, up_up = batch_bounds(batch_upper_reach(batch_reach), j)
#     low_low, low_up = batch_bounds(batch_lower_reach(batch_reach), j)


#     up_slope = relaxed_relu_gradient(up_low, up_up)
#     low_slope = relaxed_relu_gradient(low_low, low_up)

#     output_Up[j, :] .*= up_slope
#     output_Up[j, end] += up_slope * max(-up_low, 0)

#     output_Low[j, :] .*= low_slope

#     LΛᵢ[j], UΛᵢ[j] = low_slope, up_slope


#     sym = SymbolicInterval(output_Low, output_Up, domain(input))
#     LΛ = push!(input.LΛ, LΛᵢ)
#     UΛ = push!(input.UΛ, UΛᵢ)
#     return SymbolicIntervalGradient(sym, LΛ, UΛ)
# end


# # Symbolic forward_act
# function forward_act(::ReluVal, L::Layer{Id}, input::SymbolicIntervalMask)
#     n_node = size(input.sym.Up, 1)
#     LΛ = push!(input.LΛ, trues(n_node))
#     UΛ = push!(input.UΛ, trues(n_node))
#     return SymbolicIntervalGradient(input.sym, LΛ, UΛ)
# end

function forward_act(prop_method::Ai2h, layer::typeof(relu), batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [convex_hull(UnionSetArray(forward_partition(layer, reach))) for reach in batch_reach]
    return batch_reach, batch_info
end

function forward_act(prop_method::Ai2z, layer::typeof(relu), batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [overapproximate(Rectification(reach), Zonotope) for reach in batch_reach]
    return batch_reach, batch_info
end  

function forward_act(prop_method::Box, layer::typeof(relu), batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [rectify(reach) for reach in batch_reach]
    return batch_reach, batch_info
end  

function forward_partition(layer::typeof(relu), reach)
    N = dim(reach)
    output = HPolytope{Float64}[]
    for h in 0:(2^N)-1
        P = Diagonal(1.0.*digits(h, base = 2, pad = N))
        orthant = HPolytope(Matrix(I - 2.0P), zeros(N))
        S = intersection(reach, orthant)
        if !isempty(S)
            push!(output, linear_map(P, S))
        end
    end
    return output
end
    
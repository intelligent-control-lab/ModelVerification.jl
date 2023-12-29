"""
    propagate_linear(prop_method::ImageStar, layer::BatchNorm, 
                     bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a batch norm layer. I.e., it 
applies the batch norm operation to the `ImageStarBound` bound. The batch norm 
operation is decomposed into two operations: centering and scaling. The 
centering operation is applied to the center of the `ImageStarBound` bound. The
scaling operation is applied to the generators of the `ImageStarBound` bound.
The resulting bound is also of type `ImageStarBound`.

## Arguments
- `prop_method` (`ImageStar`): The `ImageStar` propagation method used for the 
    verification problem.
- `layer` (`BatchNorm`): The batch norm operation to be used for propagation.
- `bound` (`ImageStarBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The batch normed bound of the output layer represented in `ImageStarBound` 
    type.
"""
function propagate_linear(prop_method::ImageStar, layer::BatchNorm, bound::ImageStarBound, batch_info)
    cen_BN = @set layer.λ = identity # copy a BN and set activation to identity

    gen_BN = @set cen_BN.β = zeros(eltype(cen_BN.β), size(cen_BN.β)) # copy a BN set β to zeros
    gen_BN = @set gen_BN.μ = zeros(eltype(cen_BN.μ), size(cen_BN.μ)) # copy a BN set μ to zeros

    new_center = cen_BN(bound.center)
    new_generators = gen_BN(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b)
end

"""
    propagate_linear(prop_method::ImageZono, layer::BatchNorm, 
                     bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a batch norm layer. I.e., it 
applies the batch norm operation to the `ImageZonoBound` bound. The batch norm 
operation is decomposed into two operations: centering and scaling. The 
centering operation is applied to the center of the `ImageZonoBound` bound. The
scaling operation is applied to the generators of the `ImageZonoBound` bound.
The resulting bound is also of type `ImageZonoBound`.

## Arguments
- `prop_method` (`ImageZono`): The `ImageZono` propagation method used for the 
    verification problem.
- `layer` (`BatchNorm`): The batch norm operation to be used for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The batch normed bound of the output layer represented in `ImageZonoBound` 
    type.
"""
function propagate_linear(prop_method::ImageZono, layer::BatchNorm, bound::ImageZonoBound, batch_info)
    cen_BN = @set layer.λ = identity # copy a BN and set activation to identity

    gen_BN = @set cen_BN.β = zeros(eltype(cen_BN.β), size(cen_BN.β)) # copy a BN set β to zeros
    gen_BN = @set gen_BN.μ = zeros(eltype(cen_BN.μ), size(cen_BN.μ)) # copy a BN set μ to zeros
    
    # cen_BN = cen_BN |> gpu
    # gen_BN = gen_BN |> gpu
    # new_center = cen_BN(bound.center |> gpu) |> cpu
    # new_generators = gen_BN(bound.generators |> gpu) |> cpu
    
    new_center = cen_BN(bound.center)
    new_generators = gen_BN(bound.generators)
    # new_generators = propagate_by_small_batch(gen_BN |> gpu, bound.generators |> gpu, sm_batch=10) |> cpu
    
    return ImageZonoBound(new_center, new_generators)
end 

"""
    propagate_linear_batch(layer::BatchNorm, batch_reach::AbstractArray, 
                           batch_info)

Propagate the `batch_reach` through a batch norm layer. I.e., it applies the 
batch norm operation to the `batch_reach`. The batch norm operation is 
decomposed into two operations: centering and scaling. This function supports 
input batch with channel dimension.

## Arguments
- `layer` (`BatchNorm`): The batch norm operation to be used for propagation.
- `batch_reach` (`AbstractArray`): The batch of input bounds.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The batch normed bound of the output layer.
"""
function propagate_linear_batch(layer::BatchNorm, batch_reach::AbstractArray, batch_info)
    β, γ, μ, σ², ϵ, momentum, affine, track_stats = layer.β, layer.γ, layer.μ, layer.σ², layer.ϵ, layer.momentum, layer.affine, layer.track_stats
    channels = size(batch_reach)[end-1] #number of channels

    β = affine ? β : 1.0
    γ = affine ? γ : 0.0
    μ = track_stats ? μ : zeros(channels)
    σ² = track_stats ? σ² : ones(channels)

    tmp_β = β .- μ ./ sqrt.(σ² .+ ϵ) .* γ
    tmp_γ = γ ./ sqrt.(σ² .+ ϵ)
    tmp_γ = reshape(tmp_γ, (channels, 1)) 
    #reshape the tmp_γ for ".* batch_reach"
    for i in 1:(ndims(batch_reach)-2)
        tmp_γ = reshape(tmp_γ, (1, size(tmp_γ)...))
    end
    batch_reach = batch_reach .* tmp_γ
    channel_dim = ndims(batch_reach) - 1 #the dim of the channel
    if (ndims(batch_reach) > 2)
        tmp_batch_reach = dropdims(sum(batch_reach, dims = Tuple(1:channel_dim-1)), dims = Tuple(1:channel_dim-1)) #first sum the 1:channel_dim-1 dims, then drop them
        batch_bias = dropdims(sum((tmp_batch_reach .* tmp_β), dims = 1), dims = 1)
    else
        batch_bias = dropdims(sum((batch_reach .* tmp_β), dims = channel_dim), dims = channel_dim)
    end
    return batch_reach, batch_bias
end

"""
    propagate_linear(prop_method::Crown, layer::BatchNorm, 
                     bound::CrownBound, batch_info)

Propagate the `CrownBound` bound through a batch norm layer. I.e., it 
applies the batch norm operation to the `CrownBound` bound. 
The resulting bound is also of type `CrownBound`.

## Arguments
- `prop_method` (`Crown`): The `Crown` propagation method used for the 
    verification problem.
- `layer` (`BatchNorm`): The batch norm operation to be used for propagation.
- `bound` (`CrownBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The batch normed bound of the output layer represented in `CrownBound` 
    type.
"""
function propagate_linear_batch(prop_method::Crown, layer::BatchNorm, bound::CrownBound, batch_info)
    # @show size(bound.batch_Low)
    β, γ, μ, σ², ϵ, momentum, affine, track_stats = layer.β, layer.γ, layer.μ, layer.σ², layer.ϵ, layer.momentum, layer.affine, layer.track_stats
    # @show size(μ)
    # @show size(σ²)
    lower_weight = bound.batch_Low[:,:, :, 1:end-1,:]
    upper_weight = bound.batch_Up[:,:, :, 1:end-1,:]
    lower_bias = bound.batch_Low[:,:,:, end,:]
    upper_bias = bound.batch_Up[:,:,:, end,:]
    input_dim = size(lower_weight)[4]
    batch_size = size(lower_weight)[5]
    width = size(lower_weight)[1]
    height = size(lower_weight)[2]
    channel = size(lower_weight)[3]
    lower_weight = reshape(lower_weight, (width,height,channel, input_dim*batch_size))
    upper_weight = reshape(upper_weight, (width,height,channel, input_dim*batch_size))

    channels = size(lower_weight)[end-1] #number of channels

    β = affine ? β : 1.0
    γ = affine ? γ : 0.0
    μ = track_stats ? μ : zeros(channels)
    σ² = track_stats ? σ² : ones(channels)

    tmp_β = β .- μ ./ sqrt.(σ² .+ ϵ) .* γ
    tmp_γ = γ ./ sqrt.(σ² .+ ϵ)


    tmp_γ = reshape(tmp_γ, (channels, 1)) 
    #reshape the tmp_γ for ".* batch_reach"
    for i in 1:(ndims(lower_weight)-2)
        tmp_γ = reshape(tmp_γ, (1, size(tmp_γ)...))
    end
    # @show size(tmp_γ)
    # @show size(lower_weight .* pos_γ)
    # @show size(lower_weight .* pos_γ)
    pos_γ = clamp.(tmp_γ, 0, Inf)
    neg_γ = clamp.(tmp_γ, -Inf, 0)
    lw = lower_weight .* pos_γ + upper_weight .* neg_γ
    uw = upper_weight .* pos_γ + lower_weight .* neg_γ

    tmp_β = reshape(tmp_β, size(tmp_γ))
    # @show size(lower_bias)
    # @show size(tmp_β)
    # @show size(lower_bias .* pos_γ)
    lb = lower_bias .* pos_γ + upper_bias .* neg_γ .+ tmp_β
    ub = upper_bias .* pos_γ + lower_bias .* neg_γ .+ tmp_β

    lw = reshape(lw, (size(lw)[1:3]...,input_dim,batch_size))
    uw = reshape(uw, (size(uw)[1:3]...,input_dim,batch_size))
    lb = reshape(lb, (size(lb)[1:3]...,1,batch_size))
    ub = reshape(ub, (size(ub)[1:3]...,1,batch_size))
    # @show size(lw)
    # @show size(cat(lw,lb, dims=4)), size(lb)
    new_bound = CrownBound(cat(lw,lb, dims=4), cat(uw,ub, dims=4), bound.batch_data_min, bound.batch_data_max, bound.img_size)
    # @show size(new_bound.batch_Low)
    return new_bound
end 
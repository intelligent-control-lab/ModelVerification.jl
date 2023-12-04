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

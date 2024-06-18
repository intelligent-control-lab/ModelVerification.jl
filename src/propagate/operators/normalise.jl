"""
    propagate_layer(prop_method::ImageStar, layer::BatchNorm, 
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
function propagate_layer(prop_method::ImageStar, layer::BatchNorm, bound::ImageStarBound, batch_info)
    cen_BN = @set layer.λ = identity # copy a BN and set activation to identity

    gen_BN = @set cen_BN.β = zeros(eltype(cen_BN.β), size(cen_BN.β)) # copy a BN set β to zeros
    gen_BN = @set gen_BN.μ = zeros(eltype(cen_BN.μ), size(cen_BN.μ)) # copy a BN set μ to zeros

    new_center = cen_BN(bound.center)
    new_generators = gen_BN(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b)
end

"""
    propagate_layer(prop_method::ImageZono, layer::BatchNorm, 
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
function propagate_layer(prop_method::ImageZono, layer::BatchNorm, bound::ImageZonoBound, batch_info)
    cen_BN = @set layer.λ = identity # copy a BN and set activation to identity

    gen_BN = @set cen_BN.β = zeros(eltype(cen_BN.β), size(cen_BN.β)) # copy a BN set β to zeros
    gen_BN = @set gen_BN.μ = zeros(eltype(cen_BN.μ), size(cen_BN.μ)) # copy a BN set μ to zeros
    
    # cen_BN = cen_BN |> gpu
    # gen_BN = gen_BN |> gpu
    # new_center = cen_BN(bound.center |> gpu) |> cpu
    # new_generators = gen_BN(bound.generators |> gpu) |> cpu
    
    new_center = FloatType[].(cen_BN(bound.center))

    n_gen = size(bound.generators, 4)
    new_generators = FloatType[].(zeros(size(new_center)[1:end-1]..., n_gen))
    bs = 256
    for i in 1:bs:n_gen
        t = min(i+bs-1, n_gen)
        if prop_method.use_gpu
            new_generators[:,:,:,i:t] = gen_BN(bound.generators[:,:,:,i:t]) |> gpu |> cpu
        else
            new_generators[:,:,:,i:t] = gen_BN(bound.generators[:,:,:,i:t])
        end
    end
    # new_generators = gen_BN(bound.generators)

    # new_generators = propagate_by_small_batch(gen_BN |> gpu, bound.generators |> gpu, sm_batch=10) |> cpu
    
    return ImageZonoBound(new_center, new_generators)
end 

"""
    propagate_layer_batch(layer::BatchNorm, batch_reach::AbstractArray, 
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
function propagate_layer_batch(layer::BatchNorm, batch_reach::AbstractArray, batch_info)
    β, γ, μ, σ², ϵ, momentum, affine, track_stats = layer.β, layer.γ, layer.μ, layer.σ², layer.ϵ, layer.momentum, layer.affine, layer.track_stats
    channels = size(batch_reach)[end-1] #number of channels

    β = affine ? β : 0.0
    γ = affine ? γ : 1.0
    μ = track_stats ? μ : zeros(channels)
    σ² = track_stats ? σ² : ones(channels)

    shift = β .- μ ./ sqrt.(σ² .+ ϵ) .* γ
    scale = γ ./ sqrt.(σ² .+ ϵ)
    scale = reshape(scale, (channels, 1)) 
    #reshape the scale for ".* batch_reach"
    for i in 1:(ndims(batch_reach)-2)
        scale = reshape(scale, (1, size(scale)...))
    end
    batch_reach = batch_reach .* scale
    channel_dim = ndims(batch_reach) - 1 #the dim of the channel
    if (ndims(batch_reach) > 2)
        tmp_batch_reach = dropdims(sum(batch_reach, dims = Tuple(1:channel_dim-1)), dims = Tuple(1:channel_dim-1)) #first sum the 1:channel_dim-1 dims, then drop them
        batch_bias = dropdims(sum((tmp_batch_reach .* shift), dims = 1), dims = 1)
    else
        batch_bias = dropdims(sum((batch_reach .* shift), dims = channel_dim), dims = channel_dim)
    end
    return batch_reach, batch_bias
end

"""
    propagate_layer(prop_method::Crown, layer::BatchNorm, 
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
function propagate_layer_batch(prop_method::Crown, layer::BatchNorm, bound::CrownBound, batch_info)
    # @show size(bound.batch_Low)
    β = prop_method.use_gpu ? fmap(cu, layer.β) : layer.β
    γ = prop_method.use_gpu ? fmap(cu, layer.γ) : layer.γ
    μ = prop_method.use_gpu ? fmap(cu, layer.μ) : layer.μ
    σ² = prop_method.use_gpu ? fmap(cu, layer.σ²) : layer.σ²
    ϵ = prop_method.use_gpu ? fmap(cu, layer.ϵ) : layer.ϵ
    momentum, affine, track_stats = layer.momentum, layer.affine, layer.track_stats
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

    β = affine ? β : 0.0
    γ = affine ? γ : 1.0
    μ = track_stats ? μ : zeros(channels)
    σ² = track_stats ? σ² : ones(channels)

    shift = β .- μ ./ sqrt.(σ² .+ ϵ) .* γ
    scale = γ ./ sqrt.(σ² .+ ϵ)

    scale = reshape(scale, (channels, 1)) 
    #reshape the scale for ".* batch_reach"
    for i in 1:(ndims(lower_weight)-2)
        scale = reshape(scale, (1, size(scale)...))
    end
    # @show size(scale)
    # @show size(lower_weight .* pos_γ)
    # @show size(lower_weight .* pos_γ)
    pos_γ = clamp.(scale, 0, Inf)
    neg_γ = clamp.(scale, -Inf, 0)
    lw = lower_weight .* pos_γ + upper_weight .* neg_γ
    uw = upper_weight .* pos_γ + lower_weight .* neg_γ

    shift = reshape(shift, size(scale))
    # @show size(lower_bias)
    # @show size(shift)
    # @show size(lower_bias .* pos_γ)
    lb = lower_bias .* pos_γ + upper_bias .* neg_γ .+ shift
    ub = upper_bias .* pos_γ + lower_bias .* neg_γ .+ shift

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


function propagate_layer_batch(prop_method::IBP, layer::BatchNorm, bound::IBPBound, batch_info)
    new_low = layer(bound.batch_low)
    new_up = layer(bound.batch_up)
    return IBPBound(new_low, new_up)
    
    β = layer.β
    γ = layer.γ
    μ = layer.μ
    σ² = layer.σ²
    ϵ = layer.ϵ
    momentum, affine, track_stats = layer.momentum, layer.affine, layer.track_stats
    
    β = affine ? β : 0.0
    γ = affine ? γ : 1.0
    μ = track_stats ? μ : zeros(channels)
    σ² = track_stats ? σ² : ones(channels)

    shift = β .- μ ./ sqrt.(σ² .+ ϵ) .* γ
    scale = γ ./ sqrt.(σ² .+ ϵ)

    @show size(shift)
    @show size(scale)
    @show size(bound.batch_low)
    @show size(bound.batch_up)

    shape = ntuple(i -> i == N-1 ? size(bound.batch_low, N-1) : 1, N)

    scale = reshape(scale, (size(scale)..., 1)) 
    shift = reshape(shift, (size(shift)..., 1)) 

    @show size(shift)
    @show size(scale)

    new_low = scale .* bound.batch_low .+ shift
    new_up = scale .* bound.batch_up .+ shift

    return IBPBound(new_low, new_up)
end 


"""
    propagate_linear(prop_method::BetaCrown, layer::BatchNorm, 
                     bound::BetaCrownBound, batch_info)

Propagate the `BetaCrownBound` bound through a batch norm layer. I.e., it 
applies the "inverse" batch norm operation to the `BetaCrownBound` bound. 
The resulting bound is also of type `BetaCrownBound`.

## Arguments
- `prop_method` (`Crown`): The `Crown` propagation method used for the 
    verification problem.
- `layer` (`BetaCrown`): The batch norm operation to be used for propagation.
- `bound` (`BetaCrownBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The batch normed bound of the output layer represented in `BetaCrownBound` 
    type.
"""
function propagate_layer_batch(prop_method::BetaCrown, layer::BatchNorm, bound::BetaCrownBound, batch_info)
    node = batch_info[:current_node]
    
    β, γ, μ, σ², ϵ, momentum, affine, track_stats = layer.β, layer.γ, layer.μ, layer.σ², layer.ϵ, layer.momentum, layer.affine, layer.track_stats
    
    channels = batch_info[node][:size_after_layer][3] #number of channels

    β = affine ? β : 0.0
    γ = affine ? γ : 1.0
    μ = track_stats ? μ : zeros(channels)
    σ² = track_stats ? σ² : ones(channels)

    shift = β .- μ ./ sqrt.(σ² .+ ϵ) .* γ # bias
    scale = γ ./ sqrt.(σ² .+ ϵ) # weight
    
    #TODO: we haven't consider the perturbation in weight and bias
    @assert !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
    lA_W = uA_W = lA_bias = uA_bias = lA_x = uA_x = nothing 
    # @show node
    size_after_layer = batch_info[node][:size_after_layer][1:3]
    size_before_layer = batch_info[node][:size_before_layer][1:3]
    # @show size_before_layer
    # @show size_after_layer
    # weight, bias, stride, pad, dilation, groups = layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups

    # bias_lb = _preprocess(node, batch_info, layer.bias)
    # bias_ub = _preprocess(node, batch_info, layer.bias)
    # lA_W = uA_W = lA_bias = uA_bias = lA_x = uA_x = nothing 
    # println("=== in cnn ===")
    # println("bound.lower_A_x: ", bound.lower_A_x)
    # if !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
    # weight = layer.weight # x[1].lower
    # bias = bias_lb # x[2].lower
    if prop_method.bound_lower
        lA_x = bn_bound_oneside(bound.lower_A_x, scale, shift, size_before_layer,size_after_layer, batch_info[:batch_size])
    else
        lA_x = nothing
    end
    if prop_method.bound_upper
        uA_x = bn_bound_oneside(bound.upper_A_x, scale, shift,size_before_layer, size_after_layer, batch_info[:batch_size])
    else
        uA_x = nothing
    end
    # println("lA_x: ", lA_x)
    # println("uA_x: ", uA_x)
    New_bound = BetaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return New_bound
    # end
end 

"""
    bn_bound_oneside(last_A, weight, bias, stride, pad, dilation, groups, size_before_layer,size_after_layer, batch_size)

"""
function bn_bound_oneside(last_A, scale, shift, size_before_layer,size_after_layer, batch_size)
    function find_w_b(x)
        # @show size_before_layer, size_after_layer
        x_weight = x[1]
        # @show size(x_weight)
        x_bias = x[2]
        # @show size(x_bias)

        x_weight = permutedims(x_weight,(2,1,3)) # spec_dim x out_dim x batch_size => #  out_dim x spec_dim x batch_size
        # @show size(x_weight)
        spec_dim = size(x_weight)[2]
        b_size = size(x_weight)[3]
        x_weight = reshape(x_weight, (size_after_layer..., spec_dim*b_size))
        # @show size(x_weight)
        @assert ndims(x_weight) > 3 # TODO: currently BN only supports CNN

        scale = reshape(scale, (size_after_layer[3], 1)) 
        #reshape the scale for ".* x_weight"
        for i in 1:((ndims(x_weight)-2))
            scale = reshape(scale, (1, size(scale)...))
        end
        # @show size(scale)
        # @show size(x_weight .* scale)

        batch_reach = x_weight .* scale

        batch_reach = reshape(batch_reach, (size(batch_reach)[1]*size(batch_reach)[2]*size(batch_reach)[3],spec_dim, b_size))
        batch_reach = permutedims(batch_reach,(2,1,3))
        
        batch_bias = sum(dropdims(sum(x_weight, dims=(1, 2)), dims=(1,2)) .* shift, dims = 1) # compute the output bias
        
        @assert size(batch_bias)[1] == 1
        batch_bias = reshape(batch_bias, (spec_dim, b_size))
        # @show size(batch_reach), size(batch_bias)
        return [batch_reach, batch_bias]
    end
    push!(last_A, find_w_b)
    return last_A
end
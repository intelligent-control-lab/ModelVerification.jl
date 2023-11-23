
function propagate_by_small_batch(f, x; sm_batch=500)
    y = nothing
    batch_size = size(x)[end]
    n_dim = length(size(x))
    for i in 1:sm_batch:batch_size
        j = min(i+sm_batch-1, batch_size)
        b = f(x[(Colon() for _ in 1:n_dim-1)..., i:j])
        y = isnothing(y) ? b : cat(y, b, dims=n_dim)
    end
    return y
end
function propagate_linear(prop_method::ImageZono, layer::Conv, bound::ImageZonoBound, batch_info)
    # copy a Conv and set activation to identity
    # println("layer.bias")
    to = get_timer("Shared")
    
    @timeit to "create_cen_conv" cen_Conv = Conv(layer.weight, layer.bias, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    @timeit to "compute_cen_conv" new_center = cen_Conv(bound.center)
    # new_center = cen_Conv(bound.center |> gpu) |> cpu
    
    # copy a Conv set bias to zeros
    # @timeit to "create_gen_conv" gen_Conv = Conv(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    @timeit to "create_gen_conv" gen_Conv = Conv(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups) |> gpu
    # println("size(bound.generators): ", size(bound.generators))
    # @timeit to "compute_gen_conv" new_generators = gen_Conv(bound.generators)
    @timeit to "compute_gen_conv" new_generators = gen_Conv(bound.generators |> gpu) |> cpu
    # @timeit to "compute_gen_conv" new_generators = propagate_by_small_batch(gen_Conv, bound.generators |> gpu) |> cpu
    
    return ImageZonoBound(new_center, new_generators)
end

function propagate_linear(prop_method::ImageStar, layer::Conv, bound::ImageStarBound, batch_info)
    # copy a Conv and set activation to identity
    cen_Conv = Conv(layer.weight, layer.bias, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups) 
    new_center = cen_Conv(bound.center)
    # copy a Conv and set activation to identity
    # gen_Conv = Conv(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    # new_generators = gen_Conv(bound.generators)
    @timeit to "create_gen_conv" gen_Conv = Conv(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups) |> gpu
    @timeit to "compute_gen_conv" new_generators = gen_Conv(bound.generators |> gpu) |> cpu
    return ImageStarBound(new_center, new_generators, bound.A, bound.b)
end


function propagate_linear(prop_method::ImageZono, layer::ConvTranspose, bound::ImageZonoBound, batch_info)
    cen_Conv = ConvTranspose(layer.weight, layer.bias, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    gen_Conv = ConvTranspose(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    new_center = cen_Conv(bound.center)
    # new_center = cen_Conv(bound.center |> gpu) |> cpu
    # println("size(bound.generators): ", size(bound.generators))
    new_generators = gen_Conv(bound.generators)
    # new_generators = propagate_by_small_batch(gen_Conv, bound.generators |> gpu) |> cpu
    return ImageZonoBound(new_center, new_generators)
end


function propagate_linear(prop_method::ImageStar, layer::ConvTranspose, bound::ImageStarBound, batch_info)
    cen_Conv = ConvTranspose(layer.weight, layer.bias, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups) 
    gen_Conv = ConvTranspose(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    new_center = cen_Conv(bound.center)
    new_generators = gen_Conv(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b)
end


function bound_onside(layer::Conv{2, 4, typeof(identity), Array{Float32, 4}, Vector{Float32}}, conv_input_size::AbstractArray, batch_reach::AbstractArray)  
    #all(isa.(batch_reach, AbstractArray)) || throw("Conv only support AbstractArray type branches.")
    weight, bias, stride, pad, dilation, groups = layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups
    #size(batch_reach) = (weight, hight, channel, batch*spec)
    batch_bias = sum(dropdims(sum(batch_reach, dims=(1, 2)), dims=(1,2)) .* bias, dims = 1) # compute the output bias
    
    weight = reverse(weight, dims=2) # flip the first two dimensions of weight(left to right)
    weight = reverse(weight, dims=1) # flip the first two dimensions of weight(upside down)
    # when (W−F+2P)%S != 0, construct the output_padding
    output_padding1 = Int(conv_input_size[1]) - (Int(size(batch_reach)[1]) - 1) * stride[1] + 2 * pad[1] - 1 - (Int(size(weight)[1] - 1) * dilation[1])
    output_padding2 = Int(conv_input_size[2]) - (Int(size(batch_reach)[2]) - 1) * stride[2] + 2 * pad[2] - 1 - (Int(size(weight)[2] - 1) * dilation[2])
    bias = zeros(size(weight)[3]) #bias need to be zero
    backward = ConvTranspose(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    batch_reach = backward(batch_reach)
    if(output_padding1 != 0 || output_padding2 != 0) #determine the output size of the ConvTranspose
        batch_reach = PaddedView(0, batch_reach, (size(batch_reach)[1] + output_padding1, size(batch_reach)[2] + output_padding2, size(batch_reach)[3], size(batch_reach)[4]))
    end
    return batch_reach, batch_bias
end  


function interval_propagate(layer::Conv{2, 4, typeof(identity), Array{Float32, 4}, Vector{Float32}}, interval, C = nothing) 
    interval_low = interval[1], interval_high = interval[2]
    weight, bias, stride, pad, dilation, groups = layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups
    mid = (interval_low + interval_high) / 2.0
    diff = (interval_high - interval_low) / 2.0
    center_propagate_layer = Conv(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    center = center_propagate_layer(mid)
    weight_abs = abs.(weight)
    bias = zeros(size(weight)[4])
    deviation_propagate_layer = Conv(weight_abs, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    deviation = deviation_propagate_layer(diff)
    upper = center + deviation
    lower = center - deviation
    return [lower, upper, nothing]
end


function bound_layer(layer::Conv{2, 4, typeof(identity), Array{Float32, 4}, Vector{Float32}}, lower_weight::AbstractArray, upper_weight::AbstractArray, lower_bias::AbstractArray, upper_bias::AbstractArray)
    weight, bias, stride, pad, dilation, groups = layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups
    mid_weight = (lower_weight .+ upper_weight) / 2.0
    mid_bias = (lower_bias .+ upper_bias) / 2.0
    diff_weight = (upper_weight .- lower_weight) / 2.0
    diff_bias = (upper_bias .- lower_bias) / 2.0
    
    weight_abs = abs.(weight)

    center_bias_layer = Conv(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    center_bias = center_bias_layer(mid_bias)
    
    bias = zeros(size(weight)[4])

    center_weight_layer = Conv(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    center_weight = center_weight_layer(mid_weight)

    deviation_weight_layer = Conv(weight_abs, bias, stride = stride, pad = pad, dilation = dilation, groups = groups)
    deviation_weight = deviation_weight_layer(diff_weight)

    deviation_bias_layer = Conv(weight_abs, bias, stride = stride, pad = pad, dilation = dilation, groups = groups)
    deviation_bias = deviation_bias_layer(diff_bias)
    
    lw = center_weight .- deviation_weight
    lb = center_bias .- deviation_bias
    uw = center_weight .+ deviation_weight
    ub = center_bias .+ deviation_bias

    return lw, lb, uw, ub
end
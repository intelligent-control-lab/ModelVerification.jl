"""
    propagate_by_small_batch(f, x; sm_batch=500)

Propagate the input `x` through `f` by small batches. This is useful when the 
input `x` is too large to fit into GPU memory.

## Arguments
- `f` (function): Function to be applied to the input `x`.
- `x` (AbstractArray): Input to be propagated through `f`.
- `sm_batch` (Int): Optional argument for the size of the small batch, default 
    is 500.

## Returns
- Output of `f` applied to the input `x`.
"""
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

"""
    propagate_layer(prop_method::ImageZono, layer::Conv, 
                     bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a convolution layer. I.e., it 
applies the convolution operation to the `ImageZonoBound` bound. The convolution 
operation is applied to both the center and the generators of the 
`ImageZonoBound` bound. Using the `Flux.Conv`, a convolutional layer is made in 
`Flux` with the given `layer` properties. While `cen_Conv` (convolutional layer 
for the center image) uses the bias, the `gen_Conv` (convolutional layer for the 
generators) does not. The resulting bound is also of type `ImageZonoBound`.

## Arguments
- `prop_method` (`ImageZono`): The `ImageZono` propagation method used for the 
    verification problem.
- `layer` (`Conv`): The convolution operation to be used for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The convolved bound of the output layer represented in `ImageZonoBound` type.
"""
# function propagate_layer(prop_method::ImageZono, layer::Conv, bound::ImageZonoBound, batch_info)
#     # copy a Conv and set activation to identity
#     # println("layer.bias")
#     to = get_timer("Shared")
    
#     @timeit to "create_cen_conv" cen_Conv = Conv(layer.weight, layer.bias, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
#     @timeit to "compute_cen_conv" new_center = cen_Conv(bound.center)
#     # new_center = cen_Conv(bound.center |> gpu) |> cpu
    
#     # copy a Conv set bias to zeros
#     # @timeit to "create_gen_conv" gen_Conv = Conv(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
#     @timeit to "create_gen_conv" gen_Conv = Conv(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups) |> gpu
#     # println("size(bound.generators): ", size(bound.generators))
#     # @timeit to "compute_gen_conv" new_generators = gen_Conv(bound.generators)
#     @timeit to "compute_gen_conv" new_generators = gen_Conv(bound.generators |> gpu) |> cpu
#     # @timeit to "compute_gen_conv" new_generators = propagate_by_small_batch(gen_Conv, bound.generators |> gpu) |> cpu
    
#     return ImageZonoBound(new_center, new_generators)
# end

function propagate_layer(prop_method::ImageZono, layer::Conv, bound::ImageZonoBound, batch_info)
    to = get_timer("Shared")
    # copy a Conv and set activation to identity
    # println("layer.bias")
    cen_Conv = Conv(layer.weight, layer.bias, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    # copy a Conv set bias to zeros
    gen_Conv = Conv(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    
    @timeit to "CNN center" new_center = cen_Conv(bound.center)
    # new_center = cen_Conv(bound.center |> gpu) |> cpu
    
    if prop_method.use_gpu
        gen_Conv = gen_Conv |> gpu
    end
    println("size(bound.generators): ", size(bound.generators))
    # cnt = Dict()
    # for i in 1:size(bound.generators,4)
    #     gen = bound.generators[:,:,:,i]
    #     # @show sum(abs.(gen) .< 1e-9), 
    #     num_1 = sum(abs.(gen) .> 1e-9)
    #     haskey(cnt, num_1) || (cnt[num_1] = 0)
    #     cnt[num_1] += 1
    # end
    # res = sort(cnt)
    # idx = [x[1] for x in res]
    # val = [x[2] for x in res]
    # @show idx
    # @show val

    n_gen = size(bound.generators, 4)
    new_generators = FloatType[].(zeros(size(new_center)[1:end-1]..., n_gen))
    bs = 256
    # @show typeof(bound.generators[:,:,:,1:1])
    # @show typeof(new_generators[:,:,:,1:1])
    # @show typeof(gen_Conv(bound.generators[:,:,:,1:1] |> gpu))
    # @show typeof(gen_Conv(bound.generators[:,:,:,1:1] |> gpu) |> cpu)

    for i in 1:bs:n_gen
        t = min(i+bs-1, n_gen)
        if prop_method.use_gpu
            @timeit to "CNN generator" new_generators[:,:,:,i:t] = gen_Conv(bound.generators[:,:,:,i:t] |> gpu) |> cpu
        else
            @timeit to "CNN generator" new_generators[:,:,:,i:t] = gen_Conv(bound.generators[:,:,:,i:t])
        end
    end
    # @timeit to "CNN gen" new_generators = gen_Conv(bound.generators)

    # new_generators = propagate_by_small_batch(gen_Conv, bound.generators |> gpu) |> cpu
    # new_generators = gen_Conv(bound.generators |> gpu) |> cpu
    # new_generators = new_generators |> cpu
    return ImageZonoBound(new_center, new_generators)
end

"""
    propagate_layer(prop_method::ImageStar, layer::Conv, 
                     bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a convolution layer. I.e., it 
applies the convolution operation to the `ImageStarBound` bound. The convolution 
operation is applied to both the center and the generators of the 
`ImageStarBound` bound. Using the `Flux.Conv`, a convolutional layer is made in 
`Flux` with the given `layer` properties. While `cen_Conv` (convolutional layer 
for the center image) uses the bias, the `gen_Conv` (convolutional layer for the 
generators) does not. The resulting bound is also of type `ImageStarBound`.

## Arguments
- `prop_method` (`ImageStar`): The `ImageStar` propagation method used for the 
    verification problem.
- `layer` (`Conv`): The convolution operation to be used for propagation.
- `bound` (`ImageStarBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The convolved bound of the output layer represented in `ImageStarBound` type.                     
"""
function propagate_layer(prop_method::ImageStar, layer::Conv, bound::ImageStarBound, batch_info)
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

"""
    forward prop for CNN, Crown, box is not using symbolic bound
"""
function propagate_layer_batch(prop_method::Crown, layer::Conv, bound::CrownBound, batch_info; box=false)
    if box
        return propagate_layer_batch_box(prop_method::Crown, layer::Conv, bound::CrownBound, batch_info)
    else
        return propagate_layer_batch_symbolic(prop_method::Crown, layer::Conv, bound::CrownBound, batch_info)
    end
    
end

"""
    propagate_layer_batch_box(prop_method::Crown, layer::Conv, 
                           bound::CrownBound, batch_info)

Propagates the bounds through the convolution layer for `Crown` solver. It operates
an convolutional transformation on the given input bound and returns the output bound.
It first concretizes the bounds and forward pro asp using `batch_interval_map` function. 
Then the bound is initalized again `CrownBound` type.

## Arguments
- `prop_method` (`Crown`): `Crown` solver used for the verification process.
- `layer` (`Dense`): Dense layer of the model.
- `bound` (`CrownBound`): Bound of the input, represented by `CrownBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `new_bound` (`CrownBound`): Bound of the output after affine transformation, 
    which is represented by `CrownBound` type.
"""
function propagate_layer_batch_box(prop_method::Crown, layer::Conv, bound::CrownBound, batch_info)
    @assert length(size(bound.batch_Low)) > 3
    img_size = size(bound.batch_Low)[1:3]
    l, u = compute_bound(bound)
    img_low = reshape(l, (img_size..., size(l)[2]))
    img_up = reshape(u, (img_size..., size(u)[2]))
    new_low, new_up = batch_interval_map_box(layer, img_low, img_up)
    batch_input = [ImageConvexHull([new_low[:,:,:,i], new_up[:,:,:,i]]) for i in size(new_low)[end]]
    new_crown_bound = init_batch_bound(prop_method, batch_input,nothing)
    return new_crown_bound
end

function propagate_layer_batch(prop_method::IBP, layer::Union{Conv, ConvTranspose}, bound::IBPBound, batch_info)
    # @show bound.batch_up - bound.batch_low
    @assert all(bound.batch_up - bound.batch_low .>= 0)
    new_low, new_up = batch_interval_map_box(layer, bound.batch_low, bound.batch_up)
    @assert all(new_up - new_low .>= 0)
    return IBPBound(new_low, new_up)
end

"""
    propagate_layer_batch_symbolic(prop_method::Crown, layer::Conv, 
                           bound::CrownBound, batch_info)

Propagates the bounds through the convolution layer for `Crown` solver. It operates
an convolutional transformation on the given input bound and returns the output bound.
It adopt symbolic forward prop using `batch_interval_map` function. 
Then the bound is initalized again `CrownBound` type.

## Arguments
- `prop_method` (`Crown`): `Crown` solver used for the verification process.
- `layer` (`Dense`): Dense layer of the model.
- `bound` (`CrownBound`): Bound of the input, represented by `CrownBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `new_bound` (`CrownBound`): Bound of the output after affine transformation, 
    which is represented by `CrownBound` type.
"""
function propagate_layer_batch_symbolic(prop_method::Crown, layer::Conv, bound::CrownBound, batch_info)
    @assert length(size(bound.batch_Low)) > 3
    new_crown_bound = batch_interval_map(prop_method, layer, bound)
    # img_size = size(bound.batch_Low)[1:3]
    # l, u = compute_bound(bound)
    # img_low = reshape(l, (img_size..., size(l)[2]))
    # img_up = reshape(u, (img_size..., size(u)[2]))
    # new_low, new_up = batch_interval_map(layer, img_low, img_up)
    # batch_input = [ImageConvexHull([new_low[:,:,:,i], new_up[:,:,:,i]]) for i in size(new_low)[end]]
    # new_crown_bound = init_batch_bound(prop_method, batch_input,nothing)
    return new_crown_bound
end

"""
    propagate_layer(prop_method::ImageZono, layer::ConvTranspose, 
                     bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a convolutional transpose layer. 
I.e., it applies the convolutional transpose operation to the `ImageZonoBound` 
bound. While a regular convolution reduces the spatial dimensions of an input, a 
convolutional transpose expands the spatial dimensions of an input.
The convolutional transpose operation is applied to both the center and 
the generators of the `ImageZonoBound` bound. Using the `Flux.ConvTranspose`, a 
convolutional tranpose layer is made in `Flux` with the given `layer` 
properties. While `cen_Conv` (convolutional transpose layer for the center 
image) uses the bias, the `gen_Conv` (convolutional transpose layer for the 
generators) does not. The resulting bound is also of type `ImageZonoBound`.

## Arguments
- `prop_method` (`ImageZono`): The `ImageZono` propagation method used for the 
    verification problem.
- `layer` (`ConvTranspose`): The convolutional transpose operation to be used 
    for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The convolved bound of the output layer represented in `ImageZonoBound` type.              
"""
function propagate_layer(prop_method::ImageZono, layer::ConvTranspose, bound::ImageZonoBound, batch_info)
    cen_Conv = ConvTranspose(layer.weight, layer.bias, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    gen_Conv = ConvTranspose(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    new_center = FloatType[].(cen_Conv(bound.center))
    # new_center = cen_Conv(bound.center |> gpu) |> cpu
    # println("size(bound.generators): ", size(bound.generators))
    
    if prop_method.use_gpu
        gen_Conv = gen_Conv |> gpu
    end

    n_gen = size(bound.generators, 4)
    new_generators = FloatType[].(zeros(size(new_center)[1:end-1]..., n_gen))
    bs = 256
    for i in 1:bs:n_gen
        t = min(i+bs-1, n_gen)
        if prop_method.use_gpu
            new_generators[:,:,:,i:t] = gen_Conv(bound.generators[:,:,:,i:t] |> gpu ) |> cpu
        else
            new_generators[:,:,:,i:t] = gen_Conv(bound.generators[:,:,:,i:t])
        end
    end
    
    # new_generators = propagate_by_small_batch(gen_Conv, bound.generators |> gpu) |> cpu
    return ImageZonoBound(new_center, new_generators)
end

"""
    propagate_layer(prop_method::ImageStar, layer::ConvTranspose, 
                     bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a convolutional transpose layer. 
I.e., it applies the convolutional transpose operation to the `ImageStarBound` 
bound. While a regular convolution reduces the spatial dimensions of an input, a 
convolutional transpose expands the spatial dimensions of an input.
The convolutional transpose operation is applied to both the center and 
the generators of the `ImageStarBound` bound. Using the `Flux.ConvTranspose`, a 
convolutional tranpose layer is made in `Flux` with the given `layer` 
properties. While `cen_Conv` (convolutional transpose layer for the center 
image) uses the bias, the `gen_Conv` (convolutional transpose layer for the 
generators) does not. The resulting bound is also of type `ImageStarBound`.

## Arguments
- `prop_method` (`ImageStar`): The `ImageStar` propagation method used for the 
    verification problem.
- `layer` (`ConvTranspose`): The convolutional transpose operation to be used 
    for propagation.
- `bound` (`ImageStarBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The convolved bound of the output layer represented in `ImageStarBound` type.                          
"""
function propagate_layer(prop_method::ImageStar, layer::ConvTranspose, bound::ImageStarBound, batch_info)
    cen_Conv = ConvTranspose(layer.weight, layer.bias, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups) 
    gen_Conv = ConvTranspose(layer.weight, false, identity; stride = layer.stride, pad = layer.pad, dilation = layer.dilation, groups = layer.groups)
    new_center = cen_Conv(bound.center)
    new_generators = gen_Conv(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b)
end

"""
    bound_onside(layer::Conv{2, 4, typeof(identity), 
                            Array{Float32, 4}, Vector{Float32}}, 
                 conv_input_size::AbstractArray, batch_reach::AbstractArray)

Transforms the batch reachable set to the input size of the convolutional layer 
using a `ConvTranspose` layer. First, it extracts the layer properties such as 
`weight`, `bias`, and `stride`. Then, it computes the output bias by summing 
over the batch reach and multiplying by the bias. Then, it flips the weights 
horizontally and vertically. Then, it computes the padding needed for the output 
based on the input size and the convolutional layer properties. Then, it creates 
a `ConvTranspose` layer with the calculated parameters and applies it to the 
batch reach. If additional padding is needed, it pads the output using the 
`PaddedView` function.

## Arguments
- `layer` (`Conv`): The convolutional layer to be used for propagation.
- `conv_input_size` (AbstractArray): The size of the input to the convolutional 
    layer.
- `batch_reach` (AbstractArray): The batch reachable set of the input to the 
    convolutional layer.

## Returns
- The batch reachable set and batch bias in dimension equal to the input size of 
    the convolutional layer.
"""
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

"""
    interval_propagate(layer::Conv{2, 4, typeof(identity), 
                                   Array{Float32, 4}, Vector{Float32}}, 
                       interval, C = nothing)

Propagates the interval bounds through a convolutional layer. This is used in 
the interval arithmetic for neural network verification, where the goal is to 
compute the range of possible output values given a range of input values, 
represented with `interval`. It applies the convolution operation with `Conv` 
to the center of the interval and the deviation of the interval.

## Arguments
- `layer` (`Conv`): The convolutional layer to be used for propagation.
- `interval` (Tuple): The interval bounds of the input to the convolutional 
    layer.
- `C` (nothing): Optional argument for the center of the interval, default is 
    nothing.

## Returns
- The interval bounds after convolution operation represented in an array of 
    [lower, upper, C = nothing].
"""
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

function batch_interval_map_box(layer::Conv, batch_low, batch_up) 
    interval_low = batch_low
    interval_high = batch_up
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
    return lower, upper
end

function batch_interval_map_box(layer::ConvTranspose, batch_low, batch_up)
    weight, bias, stride, pad, dilation, groups = layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups
    mid = (batch_low + batch_up) / 2.0
    diff = (batch_up - batch_low) / 2.0
    center_propagate_layer = ConvTranspose(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    center = center_propagate_layer(mid)
    weight_abs = abs.(weight)
    deviation_propagate_layer = ConvTranspose(weight_abs, zero(bias), identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    deviation = deviation_propagate_layer(diff)
    upper = center + deviation
    lower = center - deviation
    return lower, upper
end

"""
    bound_layer(layer::Conv{2, 4, typeof(identity), 
                            Array{Float32, 4}, Vector{Float32}}, 
                lower_weight::AbstractArray, upper_weight::AbstractArray, 
                lower_bias::AbstractArray, upper_bias::AbstractArray)

Propagates the bounds of weight and bias through a convolutional layer. It 
applies the convolution operation with `Conv` to the weight and bias bounds:
`upper_weight`, `lower_weight`, `upper_bias`, and `lower_bias`. 

## Arguments
- `layer` (`Conv`): The convolutional layer to be used for propagation.
- `lower_weight` (AbstractArray): The lower bound of the weight.
- `upper_weight` (AbstractArray): The upper bound of the weight.
- `lower_bias` (AbstractArray): The lower bound of the bias.
- `upper_bias` (AbstractArray): The upper bound of the bias.

## Returns
- The bounds of the weight and bias after convolution operation represented in 
    a tuple of [lower_weight, lower_bias, upper_weight, upper_bias].
"""
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

function batch_interval_map(prop_method::Crown, layer::Conv, bound::CrownBound) 
    weight = prop_method.use_gpu ? fmap(cu, layer.weight) : layer.weight
    bias = prop_method.use_gpu ? fmap(cu, layer.bias) : layer.bias
    stride, pad, dilation, groups = layer.stride, layer.pad, layer.dilation, layer.groups
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
    # @show size(lower_weight), size(lower_bias)
    
    mid_weight = (lower_weight .+ upper_weight) / 2.0
    mid_bias = (lower_bias .+ upper_bias) / 2.0
    diff_weight = (upper_weight .- lower_weight) / 2.0
    diff_bias = (upper_bias .- lower_bias) / 2.0
    # @show size(lower_weight),(lower_bias[end])
    weight_abs = abs.(weight)

    center_bias_layer = Conv(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    center_bias = center_bias_layer(mid_bias)

    # bias = zeros(size(weight)[4])
    bias = zeros(size(bias))

    center_weight_layer = Conv(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    center_weight = center_weight_layer(mid_weight)

    deviation_weight_layer = Conv(weight_abs, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    deviation_weight = deviation_weight_layer(diff_weight)

    deviation_bias_layer = Conv(weight_abs, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    deviation_bias = deviation_bias_layer(diff_bias)
    
    lw = center_weight .- deviation_weight
    lb = center_bias .- deviation_bias
    uw = center_weight .+ deviation_weight
    ub = center_bias .+ deviation_bias
    lw = reshape(lw, (size(lw)[1:3]...,input_dim,batch_size))
    uw = reshape(uw, (size(uw)[1:3]...,input_dim,batch_size))
    lb = reshape(lb, (size(lb)[1:3]...,1,batch_size))
    ub = reshape(ub, (size(ub)[1:3]...,1,batch_size))
    # @show size(lw)
    # @show size(cat(lw,lb, dims=4)), size(lb)
    new_bound = CrownBound(cat(lw,lb, dims=4), cat(uw,ub, dims=4), bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return new_bound
end

"""
    propagate_layer(prop_method::Crown, layer::ConvTranspose, 
                     bound::CrownBound, batch_info)

Propagate the `CrownBound` bound through a convolutional transpose layer. 
I.e., it applies the convolutional transpose operation to the `CrownBound` 
bound. While a regular convolution reduces the spatial dimensions of an input, a 
convolutional transpose expands the spatial dimensions of an input.
 Using the `Flux.ConvTranspose`, a 
convolutional tranpose layer is made in `Flux` with the given `layer` 
properties. The resulting bound is also of type `CrownBound`.

## Arguments
- `prop_method` (`Crown`): The `Crown` propagation method used for the 
    verification problem.
- `layer` (`ConvTranspose`): The convolutional transpose operation to be used 
    for propagation.
- `bound` (`CrownBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The convolved bound of the output layer represented in `CrownBound` type.              
"""

function propagate_layer_batch(prop_method::Crown, layer::ConvTranspose, bound::CrownBound, batch_info)
    weight = prop_method.use_gpu ? fmap(cu, layer.weight) : layer.weight
    bias = prop_method.use_gpu ? fmap(cu, layer.bias) : layer.bias
    stride, pad, dilation, groups = layer.stride, layer.pad, layer.dilation, layer.groups
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
    # @show size(lower_weight), size(lower_bias)
    
    mid_weight = (lower_weight .+ upper_weight) / 2.0
    mid_bias = (lower_bias .+ upper_bias) / 2.0
    diff_weight = (upper_weight .- lower_weight) / 2.0
    diff_bias = (upper_bias .- lower_bias) / 2.0
    # @show size(lower_weight),(lower_bias[end])
    weight_abs = abs.(weight)

    center_bias_layer = ConvTranspose(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    center_bias = center_bias_layer(mid_bias)
    # @show size(weight),size(bias)
    
    bias = zeros(size(bias))

    center_weight_layer = ConvTranspose(weight, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    center_weight = center_weight_layer(mid_weight)

    deviation_weight_layer = ConvTranspose(weight_abs, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    deviation_weight = deviation_weight_layer(diff_weight)

    deviation_bias_layer = ConvTranspose(weight_abs, bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
    deviation_bias = deviation_bias_layer(diff_bias)
    
    lw = center_weight .- deviation_weight
    lb = center_bias .- deviation_bias
    uw = center_weight .+ deviation_weight
    ub = center_bias .+ deviation_bias
    lw = reshape(lw, (size(lw)[1:3]...,input_dim,batch_size))
    uw = reshape(uw, (size(uw)[1:3]...,input_dim,batch_size))
    lb = reshape(lb, (size(lb)[1:3]...,1,batch_size))
    ub = reshape(ub, (size(ub)[1:3]...,1,batch_size))
    # @show size(lw)
    # @show size(cat(lw,lb, dims=4)), size(lb)
    new_bound = CrownBound(cat(lw,lb, dims=4), cat(uw,ub, dims=4), bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return new_bound
end

"""
    propagate_layer_batch(prop_method::BetaCrown, layer::Conv, 
                           bound::BetaCrownBound, batch_info)

Propagates the bounds through the Conv layer for `BetaCrown` solver. It 
operates an conv transformation on the given input bound and returns the
output bound. It first preprocesses the lower- and upper-bounds of the bias of 
the node using `_preprocess`. Then, it computes the interval map of the 
resulting lower- and upper-bounds using `conv_bound_oneside` function. The 
resulting bound is represented by `BetaCrownBound` type.

## Arguments
- `prop_method` (`BetaCrown`): `BetaCrown` solver used for the verification 
    process.
- `layer` (`Conv`): Conv layer of the model.
- `bound` (`BetaCrownBound`): Bound of the input, represented by 
    `BetaCrownBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `New_bound` (`BetaCrownBound`): Bound of the output after affine 
    transformation, which is represented by `BetaCrownBound` type.
"""
function propagate_layer_batch(prop_method::BetaCrown, layer::Conv, bound::BetaCrownBound, batch_info)
    node = batch_info[:current_node]
    #TODO: we haven't consider the perturbation in weight and bias
    @assert !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
    
    # @show node
    size_after_conv = batch_info[node][:size_after_layer][1:3]
    size_before_conv = batch_info[node][:size_before_layer][1:3]
    # @show size_before_conv
    weight = prop_method.use_gpu ? fmap(cu, layer.weight) : layer.weight
    # bias = prop_method.use_gpu ? fmap(cu, layer.bias) : layer.bias #
    stride, pad, dilation, groups = layer.stride, layer.pad, layer.dilation, layer.groups

    # TODO: check bias == bias_lb?
    bias_lb = _preprocess(prop_method, node, batch_info, layer.bias)
    bias_ub = _preprocess(prop_method, node, batch_info, layer.bias)
    lA_W = uA_W = nothing 
    # println("=== in cnn ===")
    # println("bound.lower_A_x: ", bound.lower_A_x)

    bias = bias_lb # x[2].lower
    
    lA_x = prop_method.bound_lower ? conv_bound_oneside(weight, bias_lb, stride, pad, dilation, groups, size_before_conv,size_after_conv, batch_info[:batch_size]) : nothing
    
    uA_x = prop_method.bound_upper ? conv_bound_oneside(weight, bias_lb, stride, pad, dilation, groups,size_before_conv, size_after_conv, batch_info[:batch_size]) : nothing

    New_bound = BetaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return New_bound
    # end
end 

"""
    conv_bound_oneside(weight, bias, stride, pad, dilation, groups, size_before_conv,size_after_conv, batch_size)

"""
function conv_bound_oneside(weight, bias, stride, pad, dilation, groups, size_before_conv,size_after_conv, batch_size)
    function bound_conv(x)
        A = x[1]
        # @show size(A)
        b = x[2]
        # @show size(b)
        # zero_bias = zeros(size(weight)[3])  #bias need to be zero
        # @show pad
        backward = ConvTranspose(weight, false, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
        A = permutedims(A,(2,1,3)) # spec_dim x out_dim x batch_size => #  out_dim x spec_dim x batch_size
        # @show size(A)
        spec_dim = size(A)[2]
        b_size = size(A)[3]
        A = reshape(A, (size_after_conv..., spec_dim*b_size))

        batch_reach = backward(A)
        # @show size(batch_reach)
        # @show size(A)[1], 
        output_padding1 = Int(size_before_conv[1]) - (Int(size(A)[1]) - 1) * stride[1] + 2 * pad[1] - 1 - (Int(size(weight)[1] - 1) * dilation[1])
        output_padding2 = Int(size_before_conv[2]) - (Int(size(A)[2]) - 1) * stride[2] + 2 * pad[2] - 1 - (Int(size(weight)[2] - 1) * dilation[2])
        # @show Int(size(A)[1]), Int(size(weight)[1])
        # @show (Int(size_before_conv[1]) + 2 * pad[1] - Int(size(weight)[1]))
        # TODO: currently we only support (i+2p-k)%s == 0 to avoid output padding, as shown in https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html, which Flux does not support
        @assert output_padding1 == 0 && output_padding2 == 0 "currently we only support (i+2p-k)%s == 0 (now =$output_padding1) to avoid output padding, which Flux does not support, as shown in https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html"
        # if(output_padding1 != 0 || output_padding2 != 0) #determine the output size of the ConvTranspose
        #     # @show output_padding1, output_padding2
        #     batch_reach = PaddedView(0, batch_reach, (size(batch_reach)[1] + output_padding1, size(batch_reach)[2] + output_padding2, size(batch_reach)[3], size(batch_reach)[4]))
        # end
        # @show size(batch_reach)
        batch_bias = sum(dropdims(sum(A, dims=(1, 2)), dims=(1,2)) .* bias, dims = 1) # compute the output bias
        
        batch_reach = reshape(batch_reach, (size(batch_reach)[1]*size(batch_reach)[2]*size(batch_reach)[3],spec_dim, b_size))
        batch_reach = permutedims(batch_reach,(2,1,3))
        @assert size(batch_bias)[1] == 1
        batch_bias = reshape(batch_bias, (spec_dim, b_size))
        # @show size(batch_reach), size(batch_bias)
        return [batch_reach, batch_bias]
    end
    return bound_conv
end

"""
propagate_layer_batch(prop_method::BetaCrown, layer::ConvTranspose, 
                           bound::BetaCrownBound, batch_info)

Propagates the bounds through the ConvTranspose layer for `BetaCrown` solver. It 
operates an ConvTranspose transformation on the given input bound and returns the
output bound. It first preprocesses the lower- and upper-bounds of the bias of 
the node using `_preprocess`. Then, it computes the interval map of the 
resulting lower- and upper-bounds using `convtrans_bound_oneside` function. The 
resulting bound is represented by `BetaCrownBound` type.

## Arguments
- `prop_method` (`BetaCrown`): `BetaCrown` solver used for the verification 
    process.
- `layer` (`ConvTranspose`): ConvTranspose layer of the model.
- `bound` (`BetaCrownBound`): Bound of the input, represented by 
    `BetaCrownBound` type.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `New_bound` (`BetaCrownBound`): Bound of the output after affine 
    transformation, which is represented by `BetaCrownBound` type.
"""
function propagate_layer_batch(prop_method::BetaCrown, layer::ConvTranspose, bound::BetaCrownBound, batch_info)
    node = batch_info[:current_node]
    #TODO: we haven't consider the perturbation in weight and bias
    @assert !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
    # @show node
    size_after_conv = batch_info[node][:size_after_layer][1:3]
    size_before_conv = batch_info[node][:size_before_layer][1:3]
    # @show size_before_conv
    weight = prop_method.use_gpu ? fmap(cu, layer.weight) : layer.weight
    # bias = prop_method.use_gpu ? fmap(cu, layer.bias) : layer.bias
    stride, pad, dilation, groups = layer.stride, layer.pad, layer.dilation, layer.groups

    bias_lb = _preprocess(prop_method, node, batch_info, layer.bias)
    bias_ub = _preprocess(prop_method, node, batch_info, layer.bias)
    lA_W = uA_W = nothing 

    bias = bias_lb # x[2].lower
    
    lA_x = prop_method.bound_lower ? convtrans_bound_oneside(weight, bias_lb, stride, pad, dilation, groups, size_before_conv,size_after_conv, batch_info[:batch_size]) : nothing    
    uA_x = prop_method.bound_upper ? convtrans_bound_oneside(weight, bias_lb, stride, pad, dilation, groups,size_before_conv, size_after_conv, batch_info[:batch_size]) : nothing

    New_bound = BetaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return New_bound
    # end
end 

"""
    convtrans_bound_oneside(weight, bias, stride, pad, dilation, groups, size_before_conv,size_after_conv, batch_size)

"""
function convtrans_bound_oneside(weight, bias, stride, pad, dilation, groups, size_before_conv,size_after_conv, batch_size)
    function bound_conv_trans(x)
        A = x[1]
        b = x[2]

        backward = Conv(weight, false, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
        A = permutedims(A,(2,1,3)) # spec_dim x out_dim x batch_size => #  out_dim x spec_dim x batch_size
        spec_dim = size(A)[2]
        b_size = size(A)[3]
        A = reshape(A, (size_after_conv..., spec_dim*b_size))

        batch_reach = backward(A)
        batch_bias = sum(dropdims(sum(A, dims=(1, 2)), dims=(1,2)) .* bias, dims = 1) # compute the output bias
        
        batch_reach = reshape(batch_reach, (size(batch_reach)[1]*size(batch_reach)[2]*size(batch_reach)[3],spec_dim, b_size))
        batch_reach = permutedims(batch_reach,(2,1,3))
        @assert size(batch_bias)[1] == 1
        batch_bias = reshape(batch_bias, (spec_dim, b_size))
        return [batch_reach, batch_bias]
    end
    return bound_conv_trans
end
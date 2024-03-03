"""
    propagate_linear(prop_method, layer::typeof(flatten), 
                     bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a flatten layer. I.e., it flattens 
the `ImageStarBound` into a `Star` type.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(flatten)`): The layer operation to be used for propagation.
- `bound` (`ImageStarBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The flattened bound of the output layer represented in `Star` type.
"""
propagate_linear(prop_method, layer::typeof(flatten), bound::ImageStarBound, batch_info) = 
    Star(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)), HPolyhedron(bound.A, bound.b))

"""
    propagate_linear(prop_method, layer::typeof(flatten), 
                     bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a flatten layer. I.e., it flattens 
the `ImageZonoBound` into a `Zonotope` type.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(flatten)`): The layer operation to be used for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The flattened bound of the output layer represented in `Zonotope` type.
"""
propagate_linear(prop_method, layer::typeof(flatten), bound::ImageZonoBound, batch_info) =
    Zonotope(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)))

"""
    propagate_linear(prop_method, layer::MeanPool, 
                     bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a mean pool layer. I.e., it applies
the mean pool operation to the `ImageStarBound` bound. The resulting bound is 
also of type `ImageStarBound`.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`MeanPool`): The mean pool operation to be used for propagation.
- `bound` (`ImageStarBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The mean pooled bound of the output layer represented in `ImageStarBound` 
    type.
"""
function propagate_linear(prop_method, layer::MeanPool, bound::ImageStarBound, batch_info)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b)
end

"""
    propagate_linear(prop_method, layer::MeanPool, 
                     bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a mean pool layer. I.e., it applies
the mean pool operation to the `ImageZonoBound` bound. The resulting bound is 
also of type `ImageZonoBound`.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`MeanPool`): The mean pool operation to be used for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The mean pooled bound of the output layer represented in `ImageZonoBound` 
    type.
"""
function propagate_linear(prop_method, layer::MeanPool, bound::ImageZonoBound, batch_info)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageZonoBound(new_center, new_generators)
end

function propagate_linear_batch(prop_method::Crown, layer::MeanPool, bound::CrownBound, batch_info,box=false)
    if box
        return propagate_linear_batch_box(prop_method, layer, bound, batch_info)
    else
        return propagate_linear_batch_symbolic(layer, bound)
    end
end

function propagate_linear_batch(prop_method::Crown, layer::typeof(Flux.flatten), bound::CrownBound, batch_info)
    bound, _ = convert_CROWN_Bound_batch(bound)
    return bound
end

function propagate_linear_batch_box(prop_method::Crown, layer::MeanPool, bound::CrownBound, batch_info)
    @assert length(size(bound.batch_Low)) > 3
    img_size = size(bound.batch_Low)[1:3]
    l, u = compute_bound(bound)
    img_low = reshape(l, (img_size..., size(l)[2]))
    img_up = reshape(u, (img_size..., size(u)[2]))
    new_low = layer(img_low)
    new_up = layer(img_up)
    batch_input = [ImageConvexHull([new_low[:,:,:,i], new_up[:,:,:,i]]) for i in size(new_low)[end]]
    new_crown_bound = init_batch_bound(prop_method, batch_input,nothing)
    return new_crown_bound
end

function propagate_linear_batch_symbolic(layer::MeanPool, bound::CrownBound)
    # width × height × channel × (input_dim+1) * batch_size
    batch_Low = reshape(bound.batch_Low, (size(bound.batch_Low)[1],size(bound.batch_Low)[2],size(bound.batch_Low)[3], size(bound.batch_Low)[4]*size(bound.batch_Low)[5]))
    batch_Up = reshape(bound.batch_Up, (size(bound.batch_Up)[1],size(bound.batch_Up)[2],size(bound.batch_Up)[3], size(bound.batch_Up)[4]*size(bound.batch_Up)[5]))
    
    new_low = layer(batch_Low)
    new_up = layer(batch_Up)
    
    
    new_low = reshape(new_low, (size(new_low)[1:3]...,size(bound.batch_Low)[4],size(bound.batch_Low)[5]))
    new_up = reshape(new_up, (size(new_up)[1:3]...,size(bound.batch_Up)[4],size(bound.batch_Up)[5]))

    # @show size(lw)
    # @show size(cat(lw,lb, dims=4)), size(lb)
    new_bound = CrownBound(new_low, new_up, bound.batch_data_min, bound.batch_data_max, bound.img_size)
    return new_bound
end

# TODO: Ad-hoc solution, needs to be replaced to improve performance.
function propagate_linear_batch(prop_method::BetaCrown, layer::MeanPool, bound::BetaCrownBound, batch_info)
    @assert all(x -> x == layer.k[1], layer.k)
    @assert all(x -> x == layer.stride[1], layer.stride)
    @assert all(x -> x == layer.pad[1], layer.pad)
    
    # Create a diagonal weight matrix for channel-wise mean pooling
    node = batch_info[:current_node]
    size_after_layer = batch_info[node][:size_after_layer][1:3]
    channel = size_after_layer[3]
    
    weights = zeros(layer.k..., channel, channel)
    v = 1 / prod(layer.k)
    for i in 1:channel
        weights[:,:,i,i] .= v
    end
    equal_conv = Conv(weights, false, identity; stride = layer.stride[1], pad = layer.pad[1])

    return propagate_linear_batch(prop_method, equal_conv, bound, batch_info)
end

""" There exist bugs in the following code, especially in function `meanpool_bound_oneside`
"""
function f_propagate_linear_batch(prop_method::BetaCrown, layer::MeanPool, bound::BetaCrownBound, batch_info)
    node = batch_info[:current_node]
    #TODO: we haven't consider the perturbation in weight and bias
    @assert !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
    lA_W = uA_W = lA_bias = uA_bias = lA_x = uA_x = nothing 
    # @show node
    size_before_layer = batch_info[node][:size_before_layer][1:3]
    # @show batch_info[node][:size_before_layer], batch_info[node][:size_after_layer]
    size_after_layer = batch_info[node][:size_after_layer][1:3]
    kernel_size = layer.k
    pad = layer.pad
    stride = layer.stride
    
    # @show size_after_layer
    # # @show layer.window
    # @show layer.k
    # @show layer.pad, layer.stride
    # weight, bias, stride, pad, dilation, groups = layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups

    # bias_lb = _preprocess(node, batch_info, layer.bias)
    # bias_ub = _preprocess(node, batch_info, layer.bias)
    # lA_W = uA_W = lA_bias = uA_bias = lA_x = uA_x = nothing 
    # println("=== in dense ===")
    # println("bound.lower_A_x: ", bound.lower_A_x)
    if !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
        # weight = layer.weight # x[1].lower
        # bias = bias_lb # x[2].lower
        if prop_method.bound_lower
            lA_x = meanpool_bound_oneside(bound.lower_A_x, kernel_size, stride, pad, bound.batch_data_min, bound.batch_data_max,size_after_layer,size_before_layer, batch_info[:batch_size])
        else
            lA_x = nothing
        end
        if prop_method.bound_upper
            uA_x = meanpool_bound_oneside(bound.upper_A_x, kernel_size, stride, pad, bound.batch_data_min, bound.batch_data_max,size_after_layer,size_before_layer, batch_info[:batch_size])
        else
            uA_x = nothing
        end
        # println("lA_x: ", lA_x)
        # println("uA_x: ", uA_x)
        New_bound = BetaCrownBound(lA_x, uA_x, lA_W, uA_W, bound.batch_data_min, bound.batch_data_max, bound.img_size)
        return New_bound
    end
end

"""
    meanpool_bound_oneside(last_A, kernel_size, stride, pad, batch_data_min, batch_data_max,size_after_layer, batch_size)

"""
function meanpool_bound_oneside(last_A, kernel_size, stride, pad, batch_data_min, batch_data_max,size_after_layer,size_before_layer, batch_size)

    #all(isa.(batch_reach, AbstractArray)) || throw("Conv only support AbstractArray type branches.")
    # weight, bias, stride, pad, dilation, groups = layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups
    #size(batch_reach) = (weight, hight, channel, batch*spec)
    # @show size(weight) # (5, 5, 1, 6)
    # @show size(bias) # (6,)
    # @show last_A
    
    # weight = reverse(weight, dims=2) # flip the first two dimensions of weight(left to right)
    # weight = reverse(weight, dims=1) # flip the first two dimensions of weight(upside down)
    # @show size(weight) 
    function find_w_b(x)
        # @show size(x[1]), size(x[2])
        # # spec: A x - b <= 0 is the safe set or unsafe set
        # A::AbstractArray{Float64, 3} # spec_dim x out_dim x batch_size
        # b::AbstractArray{Float64, 2} # spec_dim x batch_size
        x_weight = x[1]
        # x_weight = cat(x_weight, x_weight, dims=(3,3))
        # @show size(x_weight)
        x_bias = zeros(size(x[2]))
        # lb, ub = Compute_bound(batch_data_min,batch_data_max)(x)
        # zero_bias = zeros(size(weight)[3])  #bias need to be zero
        # backward = ConvTranspose(weight, zero_bias, identity, stride = stride, pad = pad, dilation = dilation, groups = groups)
        x_weight = permutedims(x_weight,(2,1,3)) # spec_dim x out_dim x batch_size => #  out_dim x spec_dim x batch_size
        spec_dim = size(x_weight)[2]
        b_size = size(x_weight)[3]
        x_weight = reshape(x_weight, (size_after_layer..., spec_dim*b_size))
        # @show x_weight
        upsampled_weight = upsample_bilinear(x_weight, kernel_size ./ prod(kernel_size), align_corners=false)
        # @show upsampled_weight
        # batch_reach = upsampled_weight
        # @show size_before_layer[1] - size(upsampled_weight)[1]
        # TODO: check dimensions of zeros in the following padding
        batch_reach = pad_constant(upsampled_weight, (Int(0.5 * (size_before_layer[1] - size(upsampled_weight)[1])), Int(0.5 * (size_before_layer[2] - size(upsampled_weight)[2])), 0, 0), 0)
        # @show size(batch_reach)
        # x_bias = reshape(x_bias, size(x_bias)[1]*size(x_bias)[2])
        # (batch_info[model_info.final_nodes[1]][:bound].img_size...,batch_size)
        # @show size(x_weight)
        # @show size(x_bias)
        # @show size(x_weight, 2)
        # batch_reach = backward(x_weight)
        # # @show size(x_weight)[1], 
        # output_padding1 = Int(size(batch_reach)[1]) - (Int(size(x_weight)[1]) - 1) * stride[1] + 2 * pad[1] - 1 - (Int(size(weight)[1] - 1) * dilation[1])
        # output_padding2 = Int(size(batch_reach)[2]) - (Int(size(x_weight)[2]) - 1) * stride[2] + 2 * pad[2] - 1 - (Int(size(weight)[2] - 1) * dilation[2])
        # if(output_padding1 != 0 || output_padding2 != 0) #determine the output size of the ConvTranspose
        #     # println("check output size")
        #     # @show output_padding1, output_padding2
        #     batch_reach = PaddedView(0, batch_reach, (size(batch_reach)[1] + output_padding1, size(batch_reach)[2] + output_padding2, size(batch_reach)[3], size(batch_reach)[4]))
        # end
        # # @show size(dropdims(sum(x_weight, dims=(1, 2)), dims=(1,2)))
        # # @show size(x_weight)
        # batch_bias = sum(dropdims(sum(x_weight, dims=(1, 2)), dims=(1,2)) .* bias, dims = 1) # compute the output bias
        
        batch_reach = reshape(batch_reach, (size(batch_reach)[1]*size(batch_reach)[2]*size(batch_reach)[3],spec_dim, b_size))
        batch_reach = permutedims(batch_reach,(2,1,3))
        # @assert size(batch_bias)[1] == 1
        # batch_bias = reshape(batch_bias, (spec_dim, b_size))
        # @show size(batch_reach)
        return [batch_reach, x_bias]
    end
    # when (W−F+2P)%S != 0, construct the output_padding
    # println("-----")
    push!(last_A, find_w_b)
    # println("=====")
    return last_A
end

function propagate_linear_batch(prop_method::BetaCrown, layer::typeof(Flux.flatten), bound::BetaCrownBound, batch_info)
    # bound, _ = convert_CROWN_Bound_batch(bound)
    node = batch_info[:current_node]
    @assert !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
    lA_W = uA_W = lA_bias = uA_bias = lA_x = uA_x = nothing 
    # @show node
    size_before_layer = batch_info[node][:size_before_layer][1:3]
    # @show batch_info[node][:size_before_layer], batch_info[node][:size_after_layer]
    size_after_layer = [batch_info[node][:size_after_layer][1]]
    # kernel_size = layer.k
    # pad = layer.pad
    # stride = layer.stride
    
    # @show size_after_layer
    # # @show layer.window
    # @show layer.k
    # @show layer.pad, layer.stride
    # weight, bias, stride, pad, dilation, groups = layer.weight, layer.bias, layer.stride, layer.pad, layer.dilation, layer.groups

    # bias_lb = _preprocess(node, batch_info, layer.bias)
    # bias_ub = _preprocess(node, batch_info, layer.bias)
    # lA_W = uA_W = lA_bias = uA_bias = lA_x = uA_x = nothing 
    # println("=== in dense ===")
    # println("bound.lower_A_x: ", bound.lower_A_x)
    # if !batch_info[node][:weight_ptb] && (!batch_info[node][:bias_ptb] || isnothing(layer.bias))
        # weight = layer.weight # x[1].lower
        # bias = bias_lb # x[2].lower
    if prop_method.bound_lower
        lA_x = flatten_bound_oneside(bound.lower_A_x,size_before_layer, size_after_layer, batch_info[:batch_size])
    else
        lA_x = nothing
    end
    if prop_method.bound_upper
        uA_x = flatten_bound_oneside(bound.upper_A_x,size_before_layer,size_after_layer, batch_info[:batch_size])
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
flatten_bound_oneside(last_A, kernel_size, stride, pad, batch_data_min, batch_data_max,size_after_layer, batch_size)

"""
function flatten_bound_oneside(last_A,size_before_layer, size_after_layer, batch_size)
    function find_w_b(x)
        x_weight = x[1]
        @assert size(x_weight)[2] == size_after_layer[1]
        x_bias = zeros(size(x[2]))
        batch_reach = x_weight
        return [batch_reach, x_bias]
    end
    push!(last_A, find_w_b)
    return last_A
end
function forward_layer(prop_method, layer, batch_bound, batch_info)
    batch_bound, batch_info = forward_linear(prop_method, layer, batch_bound, batch_info)
    hasfield(typeof(layer), :σ) && (batch_bound, batch_info = forward_act(prop_method, layer.σ, batch_bound, batch_info))
    return batch_bound, batch_info
end

function backward_layer(prop_method, layer, batch_reach, batch_info)
    batch_reach, batch_info = backward_linear(prop_method, layer, batch_reach, batch_info)
    hasfield(typeof(layer), :σ) && (batch_reach, batch_info = backward_act(prop_method, layer.σ, batch_reach, batch_info))
    return batch_reach, batch_info
end

function propagate(prop_method::ForwardProp, model, batch_bound, batch_out_spec, batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    
    for layer in model.layers
        if isa(layer, SkipConnection)
            batch_bound, batch_info = propagate(prop_method, layer.layers, batch_bound, batch_out_spec, batch_info)
        else
            batch_bound, batch_info = forward_layer(prop_method, layer, batch_bound, batch_info)
            #println(layer, " reach ", low(batch_bound[1]), high(batch_bound[1]))
        end
    end

    return batch_bound, batch_info
end

function propagate(prop_method::BackwardProp, model, batch_bound, batch_out_spec, batch_info)
    # output: batch x ... x ...
    # dfs start from model.output_nodes
    
    input_size = forward(model, batch_input) #using forward to get the input size of the layers in the model
    layer_index = 0
    for layer in reverse(model.layers)
        if isa(layer, SkipConnection)
            batch_bound, batch_bias, batch_info = propagate(prop_method, layer.layers, batch_input, batch_bound, batch_info)
        elseif isa(layer, Conv)
            conv_input_size = input_size[end-layer_index] #get the original forward input size 
            batch_bound, batch_bias, batch_info = backward_linear(layer, conv_input_size, batch_bound, batch_info)
        end
        layer_index += 1
    end
    return batch_bound, batch_info
end

function propagate(prop_method::AdversarialAttack, model, batch_input, batch_out_spec, batch_info)
    # output: batch x ... x ...
    throw("unimplemented")
    # couterexample_result, batch_info = attack(prop_method, model, batch_input, batch_out_spec, batch_info)
    # return couterexample_result, batch_info
end

function forward(model, batch_input::AbstractArray)
    input_size = [] #input_size is a list
    for layer in (model.layers)
        layer_input_size = [size(batch_input, i) for i in 1:ndims(batch_input)-2] #only get the input size of the layer, the last 2 dims if channel and batchsize  
        push!(input_size, layer_input_size)
        batch_input = layer(batch_input)
    end
    return input_size
end

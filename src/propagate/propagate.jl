

abstract type ForwardProp <: PropMethod end
abstract type BackwardProp <: PropMethod end
abstract type AdversarialAttack <: PropMethod end


function propagate(prop_method::ForwardProp, model, batch_input, batch_output, batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    batch_reach = batch_input
    for layer in model.layers
        if isa(layer, Dense)
            batch_reach, batch_info = forward_layer(prop_method, layer, batch_reach, batch_info)
            #println(layer, " reach ", low(batch_reach[1]), high(batch_reach[1]))
        elseif isa(layer, SkipConnection)
            batch_reach, batch_info = propagate(prop_method, layer.layers, batch_reach, batch_output, batch_info)
        end
    end

    return batch_reach, batch_info
end

function propagate(prop_method::BackwardProp, model, batch_input, batch_output, batch_info)
    # output: batch x ... x ...
    # dfs start from model.output_nodes
    batch_reach = batch_output
    input_size = forward(model, batch_input) #using forward to get the input size of the layers in the model
    layer_index = 0
    for layer in reverse(model.layers)
        if isa(layer, Conv)
            conv_input_size = input_size[end-layer_index] #get the original forward input size 
            batch_reach, batch_bias, batch_info = backward_linear(layer, conv_input_size, batch_reach, batch_info)
        elseif isa(layer, SkipConnection)
            batch_reach, batch_bias, batch_info = propagate(prop_method, layer.layers, batch_input, batch_reach, batch_info)
        end
        layer_index += 1
    end
    return batch_reach, batch_info
end

function propagate(prop_method::AdversarialAttack, model, batch_input, batch_output, batch_info)
    # output: batch x ... x ...
    throw("unimplemented")
    # couterexample_result, batch_info = attack(prop_method, model, batch_input, batch_output, batch_info)
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
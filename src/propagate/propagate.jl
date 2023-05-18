

abstract type ForwardProp <: PropMethod end
abstract type BackwardProp <: PropMethod end
abstract type AdversarialAttack <: PropMethod end


function propagate(prop_method::ForwardProp, model, batch_input, batch_output, batch_info)
    # input: batch x ... x ...

    # dfs start from model.input_nodes
    batch_reach = batch_input
    for layer in model.layers
        if isa(layer, Dense)
            println("batch_reach")
            println(batch_reach)
            println(typeof(batch_reach))
            batch_reach, batch_info = forward_layer(prop_method, layer, batch_reach, batch_info)
        elseif isa(layer, SkipConnection)
            batch_reach, batch_info = propagate(prop_method, layer.layers, batch_reach, batch_output, batch_info)
        end
    end

    return check_inclusion(prop_method, model, batch_input, batch_reach, batch_output), batch_info
end

function propagate(prop_method::BackwardProp, model, batch_input, batch_output, batch_info)
    # output: batch x ... x ...
    
    # dfs start from model.output_nodes
    return back_reach, info
    return check_backward(batch_back_reach, batch_input, batch_output), batch_info
end

function propagate(prop_method::AdversarialAttack, model, batch_input, batch_output, batch_info)
    # output: batch x ... x ...
    throw("unimplemented")
    # couterexample_result, batch_info = attack(prop_method, model, batch_input, batch_output, batch_info)
    # return couterexample_result, batch_info
end

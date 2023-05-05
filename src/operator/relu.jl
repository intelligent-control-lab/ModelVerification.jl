function forward_layer(layer::ReLU, forward_reach, prop_method, info)
    forward_reach, info = forward_act(prop_method, layer, forward_reach, info)
    return forward_reach, info
end

function forward_act(prop_method::Ai2h, layer::ReLU, forward_reach::AbstractPolytope, info)
    return forward_reach, info
end

function forward_act(prop_method::Ai2z, layer::ReLU, forward_reach::AbstractPolytope, info)
    return forward_reach, info
end
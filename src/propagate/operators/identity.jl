function forward_act(prop_method::ForwardProp, σ::typeof(identity), batch_reach::Vector{<:AbstractPolytope}, batch_info)
    return batch_reach, batch_info
end
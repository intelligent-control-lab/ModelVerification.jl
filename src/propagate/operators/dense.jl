


# Ai2z, Ai2h
function forward_linear(prop_method::ForwardProp, layer::Dense, batch_reach::AbstractArray, batch_info)
    all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
    batch_reach = identity.(batch_reach) # identity. converts Vector{Any} to Vector{AbstractPolytope}
    batch_reach, batch_info = affine_map(layer, batch_reach), batch_info
    return batch_reach, batch_info
end  

# Ai2 Box
function forward_linear(prop_method::Box, layer::Dense, batch_reach::AbstractArray, batch_info)
    all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
    batch_reach = identity.(batch_reach) # identity. converts Vector{Any} to Vector{AbstractPolytope}
    batch_reach, batch_info = approximate_affine_map(layer, batch_reach), batch_info
    return batch_reach, batch_info
end  

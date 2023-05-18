
function forward_layer(prop_method::Ai2h, layer::Dense, batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach, batch_info = affine_map(layer, batch_reach), batch_info
    return batch_reach, batch_info
end  

function forward_layer(prop_method::Ai2z, layer::Dense, batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach, batch_info = affine_map(layer, batch_reach), batch_info
    return batch_reach, batch_info
end  

function forward_layer(prop_method::Box, layer::Dense, batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach, batch_info = approximate_affine_map(layer, batch_reach), batch_info
    return batch_reach, batch_info
end  

# function forward_layer(layer::Dense, batch_reach::Vector{<:AbstractPolytope}, prop_method::Union{Ai2z, Box}, batch_info)
#     batch_reach, batch_info = forward_layer(layer, overapproximate(batch_reach, Hyperrectangle), prop_method, batch_info)
#     return batch_reach, batch_info
# end  
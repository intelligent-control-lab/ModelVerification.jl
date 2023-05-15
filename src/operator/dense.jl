include("/home/verification/ModelVerification.jl/src/operator/utils.jl")

function forward_layer(layer::Dense, forward_reach::AbstractPolytope, prop_method::Ai2h, info)
    forward_reach, info = affine_map(layer, forward_reach), nothing
    return forward_reach, info
end  

function forward_layer(layer::Dense, forward_reach::AbstractPolytope, prop_method::Ai2z, info)
    forward_reach, info = affine_map(layer, forward_reach), nothing
    return forward_reach, info
end  

function forward_layer(layer::Dense, forward_reach::AbstractPolytope, prop_method::Box, info)
    forward_reach, info = approximate_affine_map(layer, forward_reach), nothing
    return forward_reach, info
end  

function forward_layer(layer::Dense, forward_reach::AbstractPolytope, prop_method::Union{Ai2z, Box}, info)
    forward_reach, info = forward_layer(layer, overapproximate(forward_reach, Hyperrectangle), prop_method, info)
    return forward_reach, info
end  
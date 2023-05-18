
struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope}} <: ForwardProp end
Ai2() = Ai2{Zonotope}()
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Box = Ai2{Hyperrectangle} 

function forward_layer(prop_method::Ai2, layer, batch_reach::AbstractArray, batch_info)
    all(isa.(batch_reach, AbstractPolytope)) || throw("Ai2 only support AbstractPolytope type branches.")
    # identity. converts Vector{Any} to Vector{AbstractPolytope}
    return forward_layer(prop_method, layer, identity.(batch_reach), batch_info)
end

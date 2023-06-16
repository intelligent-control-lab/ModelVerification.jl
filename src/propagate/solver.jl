abstract type ForwardProp <: PropMethod end
abstract type BackwardProp <: PropMethod end
abstract type AdversarialAttack <: PropMethod end

struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope}} <: ForwardProp end
Ai2() = Ai2{Zonotope}()
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Box = Ai2{Hyperrectangle}  

struct Crown <: ForwardProp end

function prepare_method(prop_method::PropMethod, model, batch_input::AbstractVector, batch_out_spec::AbstractVector, batch_info)
    return init_bound(prop_method, batch_input), batch_out_spec, batch_info
end

function prepare_method(prop_method::Crown, model, batch_input::AbstractVector, batch_out_spec::AbstractVector, batch_info)
    return init_bound(prop_method, batch_input), get_linear_spec(batch_out_spec), batch_info
end

abstract type ForwardProp <: PropMethod end
abstract type BackwardProp <: PropMethod end
abstract type AdversarialAttack <: PropMethod end

struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope, Star}} <: ForwardProp end
Ai2() = Ai2{Zonotope}()
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Ai2s = Ai2{Star}
const Box = Ai2{Hyperrectangle}  

struct Crown <: ForwardProp end

struct ImageStar{T<:Union{Star, Zonotope}} <: ForwardProp end
ImageStar() = ImageStar{Star}()
const ImageStarZono = ImageStar{Zonotope}

function prepare_method(prop_method::PropMethod, model, batch_input::AbstractVector, batch_output::AbstractVector, batch_info)
    return init_batch_bound(prop_method, batch_input), batch_output, batch_info
end

function prepare_method(prop_method::Crown, model, batch_input::AbstractVector, batch_output::AbstractVector, batch_info)
    return init_batch_bound(prop_method, batch_input), get_linear_spec(batch_output), batch_info
end
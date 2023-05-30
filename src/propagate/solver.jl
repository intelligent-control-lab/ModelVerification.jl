
struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope}} <: ForwardProp end
Ai2() = Ai2{Zonotope}()
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Box = Ai2{Hyperrectangle}  

struct Abcrown <: BackwardProp end
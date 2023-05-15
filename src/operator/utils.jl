include("/home/verification/ModelVerification.jl/src/utils/activation.jl")
include("/home/verification/ModelVerification.jl/src/utils/network.jl")
using Interpolations
using Flux
using Polyhedra
using LazySets

"""
    affine_map(layer, x)

Compute W*x ⊕ b for a vector or LazySet `x`
"""
affine_map(layer::Dense, x) = layer.weight*x + layer.bias
function affine_map(layer::Dense, x::LazySet)
    LazySets.affine_map(layer.weight, x, layer.bias)
end

"""
   approximate_affine_map(layer, input::Hyperrectangle)

Returns a Hyperrectangle overapproximation of the affine map of the input.
"""
function approximate_affine_map(layer::Dense, input::Hyperrectangle)
    c = affine_map(layer, input.center)
    r = abs.(layer.weight) * input.radius
    return Hyperrectangle(c, r)
end

function convex_hull(U::UnionSetArray{<:Any, <:HPolytope})
    tohrep(VPolytope(LazySets.convex_hull(U)))
end
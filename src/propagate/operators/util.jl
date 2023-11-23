
"""
    affine_map(layer, x)

Compute W*x ⊕ b for a vector or LazySet `x`
"""
affine_map(layer::Dense, x::AbstractMatrix) = layer.weight*x + layer.bias

affine_map(layer::Dense, x::LazySet) = LazySets.affine_map(layer.weight, x, layer.bias)

function affine_map(layer::Dense, x::HPolytope)
    # There is a bug in CDDLib, which throws segment fault sometimes 
    # when translate HPolytope. So we have to convert it to VPolytope first
    x = tovrep(x)
    x = LazySets.affine_map(layer.weight, x, layer.bias)
    return x
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


"""
    broadcast_mid_dim(m::AbstractArray{2}, target::AbstractArray{T,3})

Given a target tensor of the shape AxBxC, 
broadcast the 2D mask of the shape AxC to AxBxC.

Outputs:
- `m` broadcasted.
"""
function broadcast_mid_dim(m::AbstractArray{T1,2}, target::AbstractArray{T2,3}) where T1 where T2
    @assert size(m,1) == size(target,1) "Size mismatch in broadcast_mid_dim"
    @assert size(m,2) == size(target,3) "Size mismatch in broadcast_mid_dim"
    # reshape the mask to match the shape of target
    m = reshape(m, size(m, 1), 1, size(m, 2)) # reach_dim x 1 x batch
    # replicate the mask along the second dimension
    m = repeat(m, 1, size(target, 2), 1)
    return m
end

struct Join{T, F}
    combine::F
    paths::T
  end
  
# allow Join(op, m1, m2, ...) as a constructor
Join(combine, paths...) = Join(combine, paths)
Flux.@functor Join
(m::Join)(x) = m.combine(map(f -> f(x), m.paths)...)
"""
    affine_map(layer::Dense, x)

Computes W*x ⊕ b for a vector `x`.

## Arguments
- `layer` (`Dense`): `Flux.Dense` layer of the model. Contains `weight` and 
    `bias`.
- `x`: Vector of values.

## Returns
- Affine mapping value: W*x ⊕ b.
"""
affine_map(layer::Dense, x::AbstractArray) = layer.weight*x + layer.bias

"""
Computes W*x ⊕ b for a LazySet `x`.

## Arguments
- `layer` (`Dense`): `Flux.Dense` layer of the model. Contains `weight` and 
    `bias`.
- `x` (`LazySet`): Node values represented with a `LazySet`.

## Returns
- Affine mapping value: W*x ⊕ b.
"""
function affine_map(layer::Dense, x::LazySet)
    return LazySets.affine_map(layer.weight, x, layer.bias)
end

function affine_map(layer::Dense, x::HPolytope)
    # There is a bug in CDDLib, which throws segment fault sometimes 
    # when translate HPolytope. So we have to convert it to VPolytope first
    x = tovrep(x) # this will convert to FloatType[]. for precision requirements.
    # println("in affine: ", eltype(x))
    x = LazySets.affine_map(layer.weight, x, layer.bias)
    return x
    affine_map(layer::Dense, x::LazySet)
end

"""
   approximate_affine_map(layer, input::Hyperrectangle)

Returns a Hyperrectangle overapproximation of the affine map of the input.

## Arguments
- `layer` (`Dense`): `Flux.Dense` layer of the model. Contains `weight` and 
    `bias`.
- `input` (`Hyperrectangle`): Node values represented with a `Hyperrectangle`.

## Returns
- Hyperrectangle overapproximation of the affine mapping value, W*x ⊕ b.
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

Given a target tensor of the shape AxBxC, broadcast the 2D mask of the shape 
AxC to AxBxC.

## Arguments
- `m` (`AbstractArray{2}`): 2D mask of shape AxC.
- `target` (`AbstractArray{T,3}`): Target tensor of the shape AxBxC.

## Returns
- `m` broadcasted to shape of `target`.
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
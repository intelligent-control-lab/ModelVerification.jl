
struct RotatedBox <: SequentialForwardProp end


"""
    compute_bound(bound::Hyperrectangle)

Computes the lower- and upper-bounds of a Hyperrectangle. 
This function is used when propagating through the layers of the model.
Radius is the sum of the absolute value of the generators of the given Hyperrectangle.

## Arguments
- `bound` (`Hyperrectangle`) : Hyperrectangle of which the bounds need to be computed

## Returns
- Lower- and upper-bounds of the Hyperrectangle.
"""
function compute_bound(bound::RotatedHyperrectangle)
    return low(bound), high(bound)
end

function init_bound(prop_method::RotatedBox, input::Hyperrectangle) 
    T = eltype(input)
    n = dim(input)
    I = Matrix{T}(LinearAlgebra.I(n))
    rb = RotatedHyperrectangle(I, input)
    return rb
end

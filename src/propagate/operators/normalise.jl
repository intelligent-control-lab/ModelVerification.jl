
function forward_linear(prop_method::ImageStar, layer::BatchNorm, bound::ImageStarBound, info)
    cen_BN = @set layer.λ = identity # copy a BN and set activation to identity

    gen_BN = @set cen_BN.β = zeros(eltype(cen_BN.β), size(cen_BN.β)) # copy a BN set β to zeros
    gen_BN = @set gen_BN.μ = zeros(eltype(cen_BN.μ), size(cen_BN.μ)) # copy a BN set μ to zeros

    new_center = cen_BN(bound.center)
    new_generators = gen_BN(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b), info
end

function forward_linear(prop_method::ImageStarZono, layer::BatchNorm, bound::ImageZonoBound, info)
    cen_BN = @set layer.λ = identity # copy a BN and set activation to identity

    gen_BN = @set cen_BN.β = zeros(eltype(cen_BN.β), size(cen_BN.β)) # copy a BN set β to zeros
    gen_BN = @set gen_BN.μ = zeros(eltype(cen_BN.μ), size(cen_BN.μ)) # copy a BN set μ to zeros

    new_center = cen_BN(bound.center)
    new_generators = gen_BN(bound.generators)
    return ImageZonoBound(new_center, new_generators), info
end

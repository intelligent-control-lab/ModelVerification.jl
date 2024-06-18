struct ExampleSolver <: SequentialForwardProp end
function init_bound(prop_method::ExampleSolver, bound::LazySet)
    h = box_approximation(bound)
    return RotatedHyperrectangle(Matrix{FloatType[]}(I, dim(h), dim(h)), h)
end
function propagate_layer(prop_method::ExampleSolver, layer::typeof(relu), reach::RotatedHyperrectangle, batch_info)
    M = Matrix{FloatType[]}(I, dim(reach), dim(reach))
    u = [LazySets.ρ(M[:,i], reach) for i in 1:dim(reach)]
    l = [LazySets.ρ(-M[:,i], reach) for i in 1:dim(reach)]
    return RotatedHyperrectangle(Matrix{FloatType[]}(I, dim(reach), dim(reach)), Hyperrectangle(low=l, high=u))
end

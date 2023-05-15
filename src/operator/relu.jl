include("/home/verification/ModelVerification.jl/src/operator/utils.jl")
include("/home/verification/ModelVerification.jl/src/operator/solver.jl")

function forward_layer(layer::ReLU, forward_reach, prop_method, info)
    forward_reach, info = forward_act(prop_method, layer, forward_reach, info)
    return forward_reach, info
end

function forward_act(prop_method::Ai2h, layer::ReLU, forward_reach::AbstractPolytope, info)
    forward_reach, info = convex_hull(UnionSetArray(forward_partition(layer, forward_reach)))
    return forward_reach, info
end

function forward_act(prop_method::Ai2z, layer::ReLU, forward_reach::AbstractPolytope, info)
    forward_reach, info = overapproximate(Rectification(forward_reach), Zonotope)
    return forward_reach, info
end  

function forward_act(prop_method::Box, layer::ReLU, forward_reach::AbstractPolytope, info)
    forward_reach, info = rectify(forward_reach)
    return forward_reach, info
end  

function forward_partition(layer::ReLU, forward_reach)
    N = dim(forward_reach)
    output = HPolytope{Float64}[]
    for h in 0:(2^N)-1
        P = Diagonal(1.0.*digits(h, base = 2, pad = N))
        orthant = HPolytope(Matrix(I - 2.0P), zeros(N))
        S = intersection(forward_reach, orthant)
        if !isempty(S)
            push!(output, linear_map(P, S))
        end
    end
    return output
end
    

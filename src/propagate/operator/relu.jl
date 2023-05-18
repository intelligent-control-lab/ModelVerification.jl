
function forward_layer(prop_method::Ai2h, layer::ReLU, batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [convex_hull(UnionSetArray(forward_partition(layer, reach))) for reach in batch_reach]
    return batch_reach, batch_info
end

function forward_layer(prop_method::Ai2z, layer::ReLU, batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [overapproximate(Rectification(reach), Zonotope) for reach in batch_reach]
    return batch_reach, batch_info
end  

function forward_layer(prop_method::Box, layer::ReLU, batch_reach::Vector{<:AbstractPolytope}, batch_info)
    batch_reach = [rectify(reach) for reach in batch_reach]
    return batch_reach, batch_info
end  

function forward_partition(layer::ReLU, reach)
    N = dim(reach)
    output = HPolytope{Float64}[]
    for h in 0:(2^N)-1
        P = Diagonal(1.0.*digits(h, base = 2, pad = N))
        orthant = HPolytope(Matrix(I - 2.0P), zeros(N))
        S = intersection(reach, orthant)
        if !isempty(S)
            push!(output, linear_map(P, S))
        end
    end
    return output
end
    

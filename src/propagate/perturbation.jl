abstract type Perturbation end

struct LP{T<:Real} <: Perturbation
    norm::Float64
    eps::Float64
end

#= function init_perturbation(node, batch_input, perturbation_info::LP, batch_info, global_info) #perturbation_info is a LP
        if(perturbation_info.norm == Inf)
            batch_Low = batch_input .- perturbation_info.eps
            batch_Up = batch_input .+ perturbation_info.eps 
        else
            batch_Low = batch_input
            batch_Up = batch_input
        end
        new_bound = CrownBound(batch_Low, batch_Up, batch_info[node]["data_min"], batch_info[node]["data_max"])
end =#

function init_perturbation(node, batch_input, perturbation_info, batch_info, global_info) #perturbation_info is a Dict()
    if(perturbation_info["norm"] == Inf)
        batch_Low = batch_input .- perturbation_info["eps"]
        batch_Up = batch_input .+ perturbation_info["eps"] 
    else
        batch_Low = batch_input
        batch_Up = batch_input
    end
    new_bound = CrownBound(batch_Low, batch_Up, batch_info[node]["data_min"], batch_info[node]["data_max"])
end

function concretize_matrix(x, A, perturbation_info, sign, batch_info, global_info)
    x_L = x - perturbation_info["eps"]
    x_U = x + perturbation_info["eps"]
    center = (x_L .+ x_U) ./ 2.0
    diff = (x_U .- x_L) ./ 2.0
    bound = NNlib.batched_mul(A, center) .+ sign .* NNlib.batched_mul(abs.(A), diff)
    return bound
end

function ptb_concretize(x, A, sign, batch_info, global_info)
    if isnothing(A)
        return nothing
    end
    return concretize_matrix(x, A, sign, batch_info, global_info)
end
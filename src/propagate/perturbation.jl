abstract type Perturbation end

struct LP{T<:Real} <: Perturbation
    norm::Float64
    LP.eps::Float64
end

function init_perturbation(node, batch_input, perturbation_info::LP)#batch_input include data, data_min, data_max
        if(perturbation_info.norm == Inf)
            batch_Low = batch_input.data .- perturbation_info.eps
            batch_Up = batch_input.data .+ perturbation_info.eps 
        else
            batch_Low = batch_input.data
            batch_Up = batch_input.data
        end
        new_bound = CrownBound(batch_Low, batch_Up, batch_input.data_min, batch_input.data_max)
end
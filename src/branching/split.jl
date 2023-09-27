@with_kw struct Bisect <: SplitMethod
    num_split::Int64     = 1
end

@with_kw struct BaBSR <: SplitMethod
    num_split::Int64     = 1
end

function split_branch(split_method::Bisect, model::Chain, input::Hyperrectangle, output, model_info, batch_info)
    #input = fmap(cu, input)
    #output = fmap(cu, output)
    split_method.num_split <= 0 && return [(input, output)]
    center, radius = LazySets.center(input), LazySets.radius_hyperrectangle(input)
    max_radius, max_idx = findmax(radius)
    input1, input2 = split_interval(input, max_idx)
    subtree1 = split_branch(Bisect(split_method.num_split-1), model, input1, output, model_info, batch_info)
    subtree2 = split_branch(Bisect(split_method.num_split-1), model, input2, output, model_info, batch_info)
    return [subtree1; subtree2]
end

function split_branch(split_method::Bisect, model::Chain, input::LazySet, output, model_info, batch_info)
    return split_branch(split_method, model, box_approximation(input), output, model_info, batch_info)
end


function split_branch(split_method::Bisect, model::Chain, input::ImageStarBound, output)
    println("splitting")
    @assert length(input.b) % 2 == 0
    n = length(input.b) ÷ 2
    T = eltype(input.A)
    I = Matrix{T}(LinearAlgebra.I(n))
    @assert all(input.A .≈ [I; .-I])
    u, l = input.b[1:n], .-input.b[n+1:end]
    max_radius, max_idx = findmax(u - l)
    bound1, bound2 = ImageStarBound(input.center, input.generators, input.A, input.b), ImageStarBound(input.center, input.generators, input.A, input.b)
    bound1.b[max_idx] = l[max_idx] + max_radius/2 # set new upper bound
    bound2.b[max_idx + n] = -(l[max_idx] + max_radius/2) # set new lower bound
    return [(bound1, output), (bound2, output)]
end


function split_branch(split_method::Bisect, model::Chain, input::ImageZonoBound, output)
    return [input, nothing] #TODO: find a way to split ImageZonoBound
end


function split_branch(split_method::Bisect, model::Chain, input::ImageStarBound, output, model_info, batch_info)
    input.A
end


function split_branch(split_method::Bisect, model::Chain, input::ImageZonoBound, output, model_info, batch_info)
    return [input, nothing] #TODO: find a way to split ImageZonoBound
end

"""
    split_interval(dom, i)

Split a set into two at the given index.

Inputs:
- `dom::Hyperrectangle`: the set to be split
- `i`: the index to split at
Return:
- `(left, right)::Tuple{Hyperrectangle, Hyperrectangle}`: two sets after split
"""

function split_interval(dom::Hyperrectangle, i::Int64)
    input_lower, input_upper = low(dom), high(dom)

    input_upper[i] = dom.center[i]
    input_split_left = Hyperrectangle(low = input_lower, high = input_upper)

    input_lower[i] = dom.center[i]
    input_upper[i] = dom.center[i] + dom.radius[i]
    input_split_right = Hyperrectangle(low = input_lower, high = input_upper)
    return (input_split_left, input_split_right)
end


#= function split_beta(S_dict, score, split_relu_node, i, split_neurons_index_in_node, j, input, output)
    # S_dict : {node => [idx_list, val_list, mask_list]}, such that we can do the following when propagate relu
    # batch_info[node][beta][S_dict[node][1]] .= S_dict[node][2]
    if i > length(split_relu_node)
        copy_S_dict = deepcopy(S_dict)
        for (idx, node) in enumerate(split_relu_node)
            copy_S_dict[node][2] = vecsign_convert_to_original_size(copy_S_dict[node][1], copy_S_dict[node][2], score[idx])
            if isnothing(copy_S_dict[node][3])
                copy_S_dict[node][3] = vecmask_convert_to_original_size(copy_S_dict[node][1], score[idx])
            else
                copy_S_dict[node][3] = copy_S_dict[node][3] .* vecmask_convert_to_original_size(copy_S_dict[node][1], score[idx])
            end
        end
        return [((input, copy_S_dict), output)]
    end
    j > length(split_neurons_index_in_node[i]) && return split_beta(S_dict, score, split_relu_node, i+1, split_neurons_index_in_node, 1, input, output)
    S_dict[split_relu_node[i]][2][j] = 1
    subtree1 = split_beta(S_dict, score, split_relu_node, i, split_neurons_index_in_node, j+1, input, output)
    S_dict[split_relu_node[i]][2][j] = -1
    subtree2 = split_beta(S_dict, score, split_relu_node, i, split_neurons_index_in_node, j+1, input, output)
    return [subtree1; subtree2]
end =#

function split_beta(S_dict, score, split_relu_node, i, split_neurons_index_in_node, j, input, output)
    # S_dict : {node => [idx_list, val_list, mask_list, history_S]}, such that we can do the following when propagate relu
    # batch_info[node][beta][S_dict[node][1]] .= S_dict[node][2]
    if i > length(split_relu_node)
        copy_S_dict = deepcopy(S_dict)
        for (idx, node) in enumerate(split_relu_node)
            copy_S_dict[node][2] = vecsign_convert_to_original_size(copy_S_dict[node][1], copy_S_dict[node][2], score[idx])
            if isnothing(copy_S_dict[node][3])
                copy_S_dict[node][3] = vecmask_convert_to_original_size(copy_S_dict[node][1], score[idx])
            else
                copy_S_dict[node][3] = copy_S_dict[node][3] .* vecmask_convert_to_original_size(copy_S_dict[node][1], score[idx])
            end
            if isnothing(copy_S_dict[node][4])
                copy_S_dict[node][4] = copy_S_dict[node][2]
            else
                copy_S_dict[node][4] .+= copy_S_dict[node][2] 
            end
        end
        return [((input, copy_S_dict), output)]
    end
    j > length(split_neurons_index_in_node[i]) && return split_beta(S_dict, score, split_relu_node, i+1, split_neurons_index_in_node, 1, input, output)
    S_dict[split_relu_node[i]][2][j] = 1
    subtree1 = split_beta(S_dict, score, split_relu_node, i, split_neurons_index_in_node, j+1, input, output)
    S_dict[split_relu_node[i]][2][j] = -1
    subtree2 = split_beta(S_dict, score, split_relu_node, i, split_neurons_index_in_node, j+1, input, output)
    return [subtree1; subtree2]
end

#= function split_branch(split_method::BaBSR, model::Chain, input::Tuple, output, model_info, batch_info)
    score = branching_scores_kfsb(model_info, batch_info, input)
    split_relu_node, split_neurons_index_in_node = topk(score, split_method.num_split, model_info)
    if length(input[2]) == 0
        S_dict = Dict(node => Any[nothing, nothing, nothing] for node in model_info.activation_nodes)
    else
        S_dict = Dict(node => Any[nothing, nothing, input[2][node][3]] for node in model_info.activation_nodes)
    end

    for (node, idx_list) in zip(split_relu_node, split_neurons_index_in_node)
        S_dict[node][1] = idx_list
        S_dict[node][2] = zeros(size(idx_list))
    end 
    return split_beta(S_dict, score, split_relu_node, 1, split_neurons_index_in_node, 1, input[1], output)#from 1st node and 1st index
end =#

function split_branch(split_method::BaBSR, model::Chain, input::Tuple, output, model_info, batch_info)
    score = branching_scores_kfsb(model_info, batch_info, input)
    split_relu_node, split_neurons_index_in_node = topk(score, split_method.num_split, model_info)
    if length(input[2]) == 0
        S_dict = Dict(node => Any[nothing, nothing, nothing, nothing] for node in model_info.activation_nodes)
    else
        S_dict = Dict(node => Any[nothing, nothing, input[2][node][3], input[2][node][4]] for node in model_info.activation_nodes)
    end

    for (node, idx_list) in zip(split_relu_node, split_neurons_index_in_node)
        S_dict[node][1] = idx_list
        S_dict[node][2] = zeros(size(idx_list))
    end 
    return split_beta(S_dict, score, split_relu_node, 1, split_neurons_index_in_node, 1, input[1], output)#from 1st node and 1st index
end

function vecsign_convert_to_original_size(index, vector, original)
    original_size_matrix = zeros(size(vec(original)))
    original_size_matrix[index] .= vector
    original_size_matrix = reshape(original_size_matrix, size(original))
    return original_size_matrix
end

function vecmask_convert_to_original_size(index, original)
    original_size_matrix = ones(size(vec(original)))
    original_size_matrix[index] .= -1
    original_size_matrix = reshape(original_size_matrix, size(original))
    return original_size_matrix
end

function branching_scores_kfsb(model_info, batch_info, input)
    score = []
    for node in model_info.activation_nodes
        if !isnothing(batch_info[node][:pre_lower_spec_A])
            #A = batch_info[node][:pre_lower_A]
            A = batch_info[node][:pre_lower_spec_A]
        else
            #A = batch_info[node][:pre_upper_A]
            A = batch_info[node][:pre_upper_spec_A]
        end
        unstable_mask = batch_info[node][:unstable_mask]
        unstable_mask = reshape(unstable_mask, (1, size(unstable_mask)...))
        lower = batch_info[node][:pre_lower]
        upper = batch_info[node][:pre_upper]
        upper_slope, upper_bias = relu_upper_bound(lower, upper)

        intercept_temp = clamp.(A, 0, Inf)
        intercept_candidate = intercept_temp .* reshape(upper_bias, (1, size(upper_bias)...))

        input_node = model_info.node_prevs[node][1]
        input_layer = model_info.node_layer[input_node]
        if isa(input_layer, Flux.Conv)
            if !isnothing(input_layer.bias)
                b_temp = input_layer.bias
            else
                b_temp = 0
            end
        elseif isa(input_layer, Flux.Dense)
            if !isnothing(input_layer.bias)
                b_temp = input_layer.bias
            else
                b_temp = 0
            end
        elseif isa(input_layer, typeof(+))
            b_temp = 0
            for l in model_info.node_prevs[input_node]
                l_layer = model_info.node_layer[l]
                if isa(l_layer, Flux.Conv)
                    if length(l_layer.inputs) > 2
                        b_temp += l_layer.bias
                    end
                end
                if isa(l_layer, Flux.normalise)
                    b_temp += 0
                end
                if isa(l_layer, typeof(+))
                    for ll in model_info.node_prevs[l]
                        ll_layer = model_info.node_layer[ll]
                        if isa(ll_layer, Flux.Conv)
                            b_temp += ll_layer.bias
                        end
                    end
                end
            end
        else
            b_temp = 0
        end   
        b_temp = fmap(cu, b_temp)
        b_temp = reshape(b_temp, (1, size(b_temp)...)) .* A
        upper_slope = reshape(upper_slope, (1, size(upper_slope)...))
        bias_candidate_1 = b_temp .* (upper_slope .- 1)
        bias_candidate_2 = b_temp .* upper_slope
        bias_candidate = min.(bias_candidate_1, bias_candidate_2)
        score_candidate = bias_candidate .+ intercept_candidate
        score_candidate = dropdims(mean((abs.(score_candidate) .* unstable_mask), dims = 1), dims = 1)
        score_candidate = mean(score_candidate, dims = ndims(score_candidate))
        #input[2] is pre_S_dict
        if length(input[2]) == 0 || isnothing(input[2][node][3])  #all relu node haven't splited || current relu node haven't splited
            splited_neurons_mask = fmap(cu, ones(size(score_candidate)))
        else
            splited_neurons_mask = fmap(cu, input[2][node][3])
        end
        score_candidate = splited_neurons_mask .* score_candidate #ensure that the neurons splitted in pre iter of propagation will not be split again
        push!(score, score_candidate)
        batch_info[node][:score_index] = length(score)
    end
    return score
end

function topk(score, k, model_info)
    vec_score = []
    relu_node_neurons_range = []
    split_relu_node = []
    split_neurons_index_in_node = []
    current_neuron_index = 1
    for matrix in score # matrix store neurons
        vec_matrix = vec(matrix)# all neurons need to be flattened into a vector 
        CUDA.@allowscalar vec_score = vcat(vec_score, vec_matrix)
        push!(relu_node_neurons_range, [current_neuron_index, current_neuron_index + length(vec_matrix) - 1])
        current_neuron_index += length(vec_matrix)
    end
    topk_index = partialsortperm(vec_score, 1:k, rev = true)
    topk_index = sort!(topk_index)
    #split_neurons_index_in_node = current_relu_node_split_neurons_index .- current_relu_node_neurons_range[1] .+ 1
    for (index, node) in enumerate(model_info.activation_nodes)
        current_relu_node_neurons_range = relu_node_neurons_range[index]
        current_relu_node_split_neurons_index = topk_index[(topk_index .>= current_relu_node_neurons_range[1]) .& (topk_index .<= current_relu_node_neurons_range[2])]
        if length(current_relu_node_split_neurons_index) != 0
            push!(split_relu_node, node)
            push!(split_neurons_index_in_node, current_relu_node_split_neurons_index .- current_relu_node_neurons_range[1] .+ 1)
        else
            continue
        end
    end
    return split_relu_node, split_neurons_index_in_node
end 
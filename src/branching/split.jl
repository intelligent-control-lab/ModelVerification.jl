@with_kw struct Bisect <: SplitMethod
    num_split::Int64     = 1
end

@with_kw struct BaBSR <: SplitMethod
    num_split::Int64     = 1
end

function split_branch(split_method::Bisect, model::Chain, input::Hyperrectangle, output)
    input = fmap(cu, input)
    output = fmap(cu, output)
    split_method.num_split <= 0 && return [(input, output)]
    center, radius = LazySets.center(input), LazySets.radius_hyperrectangle(input)
    max_radius, max_idx = findmax(radius)
    input1, input2 = split_interval(input, max_idx)
    subtree1 = split_branch(Bisect(split_method.num_split-1), model, input1, output)
    subtree2 = split_branch(Bisect(split_method.num_split-1), model, input2, output)
    return [subtree1; subtree2]
end

function split_branch(split_method::Bisect, model::Chain, input::LazySet, output)
    return split_branch(split_method, model, box_approximation(input), output)
end

function split_branch(split_method::Bisect, model::Chain, input::ImageStarBound, output)
    input.A
    
end


function split_branch(split_method::Bisect, model::Chain, input::ImageZonoBound, output)
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


function split_beta(S_dict, topk_neuron_node, i, topk_neurons_index, j, batch_info, input, output)
    # S_dict : {node => [idx_list, val_list]}, such that we can do the following when propagate relu
    # batch_info[node][beta][S_dict[node][0]] .= S_dict[node][1]
    i > length(topk_neuron_node) && return [(input, copy(S_dict)), output]
    j > length(topk_neurons_index[i]) && return split_beta(S_dict, topk_neuron_node, i+1, topk_neurons_index, 0, batch_info, input, output)
    S_dict[topk_neuron_node[i]][1][j] = -1
    S_dict[topk_neuron_node[i]][1][j] = 1
    subtree1 = split_beta(S_dict, topk_neuron_node, i, topk_neurons_index, j+1, batch_info, input, output)
    subtree2 = split_beta(S_dict, topk_neuron_node, i, topk_neurons_index, j+1, batch_info, input, output)
    return [subtree1; subtree2]
end

function split_branch(split_method::BaBSR, model::Chain, input::Hyperrectangle, output)
    score = branching_scores_kfsb(model_info, batch_info)
    topk_neuron_node, topk_neurons_index = topk(score, split_method.num_split)
    for (i, node) in enumerate(topk_neuron_node)
        if bound_lower
            batch_info[node][:beta_lower_index] = topk_neurons_index[i]
        end
        if bound_upper
            batch_info[node][:beta_upper_index] = topk_neurons_index[i]
        end
    end
    S_dict = Dict([(node, [idx_list, zeros(size(idx_list))]) for (node, idx_list) in (topk_neuron_node, topk_neurons_index)])
    return split_beta(S_dict, topk_neuron_node, i, topk_neurons_index, j, batch_info, input, output)
end


function relu_upper_bound(lower, upper)
    lower_r = clamp.(lower, -Inf, 0)
    upper_r = clamp.(upper, 0, Inf)
    upper_r .= max.(upper_r, lower_r .+ 1e-8)
    upper_slope = upper_r ./ (upper_r .- lower_r) #the slope of the relu upper bound
    upper_bias = - lower_r .* upper_slope #the bias of the relu upper bound
    return upper_slope, upper_bias
end


function branching_scores_kfsb(model_info, batch_info)
    score = []
    for node in reverse(model_info.activation_nodes)
        if !isnothing(batch_info[node][:pre_lower_A])
            A = batch_info[node][:pre_lower_A]
        else
            A = batch_info[node][:pre_upper_A]
        end
        layer = model_info.node_layer[node]
        unstable_mask = batch_info[node][:unstable_mask]
        unstable_mask = reshape(unstable_mask, (1, size(unstable_mask)...))
        lower = batch_info[node][:pre_lower]
        upper = batch_info[node][:pre_upper]
        upper_slope, upper_bias = relu_upper_bound(lower, upper)

        intercept_temp = clamp.(A, -Inf, 0)
        intercept_candidate = intercept_temp .* reshape(upper_bias, (1, size(upper_bias)...))

        input_node = model_info.pre_layer[node][1]
        input_layer = model_info.node_layer[input_node]
        if isa(layer, Flux.Conv)
            if !isnothing(input_layer.bias)
                b_temp = input_layer.bias
            else
                b_temp = 0
            end
        elseif isa(layer, Flux.Dense)
            if !isnothing(input_layer.bias)
                b_temp = input_layer.bias
            else
                b_temp = 0
            end
        elseif isa(layer, +)
            b_temp = 0
            for l in model_info.pre_layer[input_node]
                l_layer = model_info.node_layer[l]
                if isa(layer, Flux.Conv)
                    if length(l_layer.inputs) > 2
                        b_temp += typeof.bias
                    end
                end
                if isa(layer, Flux.normalise)
                    b_temp += 0
                end
                if isa(layer, +)
                    for ll in model_info.pre_layer[l]
                        ll_layer = model_info.node_layer[ll]
                        if isa(layer, Flux.Conv)
                            b_temp += ll_layer.bias
                        end
                    end
                end
            end
        else
            b_temp = 0
        end
        b_temp = reshape(b_temp, (1, size(b_temp)...)) .* A
        bias_candidate_1 = b_temp .* (upper_slope .- 1)
        bias_candidate_2 = b_temp .* upper_slope
        bias_candidate = max.(bias_candidate_1, bias_candidate_2)
        score_candidate = bias_candidate .+ intercept_candidate
        score_candidate = dropdims(mean((abs.(score_candidate) .* unstable_mask), dims = 1), dims = 1)
        push!(score, score_candidate)
        batch_info[node][:score_index] = length(score)
    end
    return score
end

function topk(score, k)
    topk_neurons = []
    topk_neurons_index = []
    for (index, matrix) in enumerate(score)
        flattened_matrix = vec(matrix)
        max_element = maximum(flattened_matrix)
        max_element_indices = findall(flattened_matrix .== max_element)
        push!(topk_neurons, (max_element, index))
        push!(topk_neurons_index, max_element_indices)
    end
    sort!(topk_neurons, by = x -> x[1], rev = true)
    sorted_indices = sortperm([x[1] for x in topk_neurons], rev = true)
    for i in 1:k
        max_element_index = sorted_indices[i]
        (max_element, matrix_index) = topk_neurons[max_element_index]
        element_indices = topk_neurons_index[matrix_index]
    end
    return topk_neuron_node, topk_neurons_index
end
#initalize relu's alpha_lower and alpha_upper
function init_alpha(layer::typeof(relu), node, batch_info)
    relu_input_lower, relu_input_upper = compute_bound(batch_info[node][:pre_bound]) # reach_dim x batch 
    #batch_size = size(relu_input_lower)[end]
    unstable_mask = (relu_input_upper .> 0) .& (relu_input_lower .< 0) #indices of non-zero alphas/ indices of activative neurons
    alpha_indices = findall(unstable_mask) 
    upper_slope, upper_bias = relu_upper_bound(relu_input_lower, relu_input_upper) #upper slope and upper bias
    lower_d = convert(typeof(upper_slope), upper_slope .> 0.5) #lower slope
    push!(batch_info[node], :alpha_shape => size(lower_d))
    #minimum_sparsity = batch_info[node]["minimum_sparsity"]
    #total_neuron_size = length(relu_input_lower) ÷ batch_size #number of the neuron of the pre_layer of relu

    #fully alpha
    @assert ndims(relu_input_lower) == 2 || ndims(relu_input_lower) == 4 "pre_layer of relu should be dense or conv"
    #if(ndims(relu_input_lower) == 2) #pre_layer of relu is dense 
    #end
    #alpha_lower is for lower bound, alpha_upper is for upper bound
    alpha_lower = alpha_upper = lower_d .* unstable_mask
    push!(batch_info[node], :alpha_lower => AlphaLayer(alpha_lower)) #reach_dim x batch
    push!(batch_info[node], :alpha_upper => AlphaLayer(alpha_upper)) #reach_dim x batch
end    


struct AlphaLayer
    alpha
end
Flux.@functor AlphaLayer

function get_lower_d(lower, upper, alpha_lower, alpha_upper)
    lower_mask = (lower .>= 0)
    upper_mask = (upper .<= 0)
    unstable_mask = (upper .> 0) .& (lower .< 0)

    if !isnothing(alpha_lower)
        lb_lower_slope = clamp.(alpha_lower, 0.0, 1.0) .* unstable_mask .+ lower_mask #the slope of unstable neuron is alpha, the slope of activative neuron is 1
    end

    if !isnothing(alpha_upper)
        ub_lower_slope = clamp.(alpha_upper, 0.0, 1.0) .* unstable_mask .+ lower_mask #the slope of unstable neuron is alpha, the slope of activative neuron is 1
    end
    
    return lb_lower_slope, ub_lower_slope
end 

#Upper bound slope and intercept according to CROWN relaxation.
function relu_upper_bound(lower, upper)
    lower_r = clamp.(lower, -Inf, 0)
    upper_r = clamp.(upper, 0, Inf)
    #lower_r .= min.(lower_r, 0)
    upper_r .= max.(upper_r, lower_r .+ 1e-8)
    upper_slope = upper_r ./ (upper_r .- lower_r) #the slope of the relu upper bound
    upper_bias = - lower_r .* upper_slope #the bias of the relu upper bound
    return upper_slope, upper_bias
end



function backward_relaxation(node, batch_info)
    if !haskey(batch_info[node], :pre_lower) || !haskey(batch_info[node], :pre_upper)
        lower, upper = compute_bound(batch_info[node][:pre_bound])
    else
        lower = batch_info[node][:pre_lower]
        upper = batch_info[node][:pre_upper]
    end
    lower_bias = [0.0]
    alpha_lower = batch_info[node][:alpha_lower]
    alpha_upper = batch_info[node][:alpha_upper]
    upper_slope, upper_bias = relu_upper_bound(lower, upper) #upper_slope:upper of slope  upper_bias:Upper of bias
    lb_lower_slope, ub_lower_slope = get_lower_d(lower, upper, alpha_lower, alpha_upper) #lower_d：lower of slope lower_d：lower of bias
    return  upper_slope, upper_bias, lb_lower_slope, ub_lower_slope, lower_bias
end 

#bound oneside of the relu, like upper or lower
function bound_oneside(last_A, slope_pos, slope_neg, bias_pos, bias_neg)
    if isnothing(last_A)
        #return None, 0
        return nothing, nothing
    end
    New_A, New_bias = multiply_by_A_signs(last_A, slope_pos, slope_neg, bias_pos, bias_neg)
    return New_A, New_bias
end

#using last_A for getting New_A
function multiply_by_A_signs(last_A, slope_pos, slope_neg, bias_pos, bias_neg)
    New_A = last_A
    New_bias = last_A
    if ndims(slope_pos) == 1
        # Special case for LSTM, the bias term is 1-dimension. 
        #New_A = clamp.(last_A, 0, Inf) .* slope_pos .+ clamp.(last_A, -Inf, 0) .* slope_neg
        #New_bias = clamp.(last_A, 0, Inf) .* bias_pos .+ clamp.(last_A, -Inf, 0) .* bias_neg
        Pos_last_A(x) = clamp.(x, 0, Inf)
        Neg_last_A(x) = clamp.(x, -Inf, 0)
        Pos_New_A = Chain(Join(.*, Pos_last_A, slope_pos))
        Neg_New_A = Chain(Join(.*, Neg_last_A, slope_neg))
        push!(New_A, Chain(Join(.*, Pos_New_A, Neg_New_A)))

        Pos_New_bias(x) = clamp.(x, 0, Inf) .* bias_pos
        Neg_New_bias(x) = clamp.(x, -Inf, 0) .* bias_neg
        push!(New_bias ,Chain(Join(.*, Pos_New_bias, Neg_New_bias)))
    else
        New_A, New_bias = clamp_mutiply(last_A, slope_pos, slope_neg, bias_pos, bias_neg)
        return New_A, New_bias
    end
end


function clamp_mutiply(last_A, slope_pos, slope_neg, bias_pos, bias_neg) 
    #A_pos = clamp.(last_A, 0, Inf)
    #A_neg = clamp.(last_A, -Inf, 0)
    #slope_pos = repeat(reshape(slope_pos,(1, size(slope_pos)...)), size(A_pos)[1], 1, 1) #add spec dim for slope_pos
    #slope_neg = repeat(reshape(slope_neg,(1, size(slope_neg)...)), size(A_neg)[1], 1, 1) #add spec dim for slope_pos
    #New_A = slope_pos .* A_pos .+ slope_neg .* A_neg 
    #new_bias_pos = new_bias_neg = [0.0]
    New_A = last_A
    New_bias = last_A

    Pos_last_A(x) = clamp.(x, 0, Inf)
    Neg_last_A(x) = clamp.(x, -Inf, 0)
    add_spec_dim(x) = repeat(reshape(x, (1, size(x)...)), batch_info[:spec_number], 1, 1) 
    Process_slope_pos = Chain([slope_pos, add_spec_dim])
    Process_slope_neg = Chain([slope_neg, add_spec_dim])
    Pos_New_A = Chain(Join(.*, Process_slope_pos, Pos_last_A))
    Neg_New_A = Chain(Join(.*, Process_slope_neg, Neg_last_A))
    New_A = Chain(Join(.*, Pos_New_A, Neg_New_A))
    if bias_pos !== nothing #new_bias_pos = torch.einsum('s...b,s...b->sb', A_pos, bias_pos)
        #= s_pos = max(size(A_pos)[end], size(bias_pos)[end])
        h_pos = max(size(A_pos)[end-1], size(bias_pos)[end-1])
        shape_A_pos = collect(size(A_pos))
        shape_b_pos = collect(size(bias_pos))

        shape_A_pos[end] = s_pos 
        shape_A_pos[end-1] = h_pos 
        shape_b_pos[end] = s_pos 
        shape_b_pos[end-1] = h_pos 

        A_pos_repeat_times = shape_A_pos .÷ collect(size(A_pos))
        b_pos_repeat_times = shape_b_pos .÷ collect(size(bias_pos))

        new_bias_pos = zeros(h_pos, s_pos)
        A_pos = repeat(A_pos, outer = A_pos_repeat_times) 
        bias_pos = repeat(bias_pos, outer = b_pos_repeat_times)
        for i in 1:s_pos
            for j in 1:h_pos
                new_bias_pos[j, i] = sum(A_pos[j, :, i] .* bias_pos[j, i])
            end
        end =#
        new_bias_pos = zeros((size(A_pos)[1], size(A_pos)[end]))#spec_dim x batch dim
        @einsum new_bias_pos[s,b] = A_pos[s,r,b] * bias_pos[r,b]
    end

    if bias_neg !== nothing #new_bias_neg = torch.einsum('...sb,...sb->sb', A_neg, bias_neg)
        new_bias_neg = zeros((size(A_neg)[1], size(A_neg)[end]))#spec_dim x batch dim
        @einsum new_bias_neg[s,b] = A_neg[s,r,b] * bias_neg[r,b]
    end
    New_bias = Chain(Join(.*, new_bias_pos, new_bias_neg))
    return New_A, New_bias
end 


function propagate_act_batch(prop_method::AlphaCrown, layer::typeof(relu), node, bound::AlphaCrownBound, batch_info)
    upper_slope, upper_bias, ub_lower_slope, lb_lower_slope, lower_bias = backward_relaxation(node, batch_info)

    if prop_method.bound_upper
        uA, ubias = bound_oneside(bound.upper_A_x, upper_slope, ub_lower_slope, upper_bias, lower_bias)
    end
    if prop_method.bound_lower
        lA, lbias = bound_oneside(bound.lower_A_x, lb_lower_slope, upper_slope, lower_bias, upper_bias)
    end
    bound = AlphaCrownBound(lA, uA, nothing, nothing, lbias, ubias, bound.batch_data_min, bound.batch_data_max)
    return bound
end
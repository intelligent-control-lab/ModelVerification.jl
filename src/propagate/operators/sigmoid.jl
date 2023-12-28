"""
    propagate_act_batch(prop_method::Crown, layer::typeof(sigmoid), 
                        bound::CrownBound, batch_info)

Propagate the `CrownBound` bound through a sigmoid layer.
"""
function propagate_act_batch(prop_method::Crown, layer::typeof(sigmoid), bound::CrownBound, batch_info)
    # println("bound sigmoid act")
    relax_lw, relax_uw, relax_lb, relax_ub = relax_bound(layer, bound)
    if length(size(bound.batch_Low)) > 3
        lower_weight = bound.batch_Low[:,:, :, 1:end-1,:]
        upper_weight = bound.batch_Up[:,:, :, 1:end-1,:]
        lower_bias = bound.batch_Low[:,:,:, end,:]
        upper_bias = bound.batch_Up[:,:,:, end,:]
        input_dim = size(lower_weight)[4]
        batch_size = size(lower_weight)[5]
        width = size(lower_weight)[1]
        height = size(lower_weight)[2]
        channel = size(lower_weight)[3]
        lower_weight = reshape(lower_weight, (width,height,channel, input_dim*batch_size))
        upper_weight = reshape(upper_weight, (width,height,channel, input_dim*batch_size))

        relax_lw = reshape(relax_lw, (width,height,channel, batch_size))
        relax_uw = reshape(relax_uw, (width,height,channel, batch_size))
        relax_lb = reshape(relax_lb, (width,height,channel, batch_size))
        relax_ub = reshape(relax_ub, (width,height,channel, batch_size))


        pos_lw = clamp.(relax_lw, 0, Inf)
        neg_lw = clamp.(relax_lw, -Inf, 0)
        new_lw = lower_weight .* pos_lw + upper_weight .* neg_lw

        pos_uw = clamp.(relax_uw, 0, Inf)
        neg_uw = clamp.(relax_uw, -Inf, 0)
        new_uw = upper_weight .* pos_uw + lower_weight .* neg_uw

        
        new_lb = lower_bias .* pos_lw + upper_bias .* neg_lw + relax_lb

        
        new_ub = upper_bias .* pos_uw + lower_bias .* neg_uw + relax_ub
        lw = reshape(new_lw, (size(new_lw)[1:3]...,input_dim,batch_size))
        uw = reshape(new_uw, (size(new_uw)[1:3]...,input_dim,batch_size))
        lb = reshape(new_lb, (size(new_lb)[1:3]...,1,batch_size))
        ub = reshape(new_ub, (size(new_ub)[1:3]...,1,batch_size))
        new_bound = CrownBound(cat(lw,lb, dims=4), cat(uw,ub, dims=4), bound.batch_data_min, bound.batch_data_max, bound.img_size)     
    else
        # for dense networks
        lower_weight = bound.batch_Low[:, 1:end-1,:]
        upper_weight = bound.batch_Up[:, 1:end-1,:]
        lower_bias = bound.batch_Low[:, end,:]
        upper_bias = bound.batch_Up[:, end,:]
        input_dim = size(lower_weight)[2]
        batch_size = size(lower_weight)[3]
        dense_size = size(lower_weight)[1]
        lower_weight = reshape(lower_weight, (dense_size, input_dim*batch_size))
        upper_weight = reshape(upper_weight, (dense_size, input_dim*batch_size))

        relax_lw = reshape(relax_lw, (dense_size, batch_size))
        relax_uw = reshape(relax_uw, (dense_size, batch_size))
        relax_lb = reshape(relax_lb, (dense_size, batch_size))
        relax_ub = reshape(relax_ub, (dense_size, batch_size))


        pos_lw = clamp.(relax_lw, 0, Inf)
        neg_lw = clamp.(relax_lw, -Inf, 0)
        new_lw = lower_weight .* pos_lw + upper_weight .* neg_lw

        pos_uw = clamp.(relax_uw, 0, Inf)
        neg_uw = clamp.(relax_uw, -Inf, 0)
        new_uw = upper_weight .* pos_uw + lower_weight .* neg_uw

        
        new_lb = lower_bias .* pos_lw + upper_bias .* neg_lw + relax_lb

        
        new_ub = upper_bias .* pos_uw + lower_bias .* neg_uw + relax_ub
        lw = reshape(new_lw, (size(new_lw)[1],input_dim,batch_size))
        uw = reshape(new_uw, (size(new_uw)[1],input_dim,batch_size))
        lb = reshape(new_lb, (size(new_lb)[1],1,batch_size))
        ub = reshape(new_ub, (size(new_ub)[1],1,batch_size))
        new_bound = CrownBound(cat(lw,lb, dims=2), cat(uw,ub, dims=2), bound.batch_data_min, bound.batch_data_max, bound.img_size)    
    end

    return new_bound
end

function relax_bound(layer::typeof(sigmoid), original_bound::CrownBound)
    # return reach_dim x batch, the same shape of concretized bounds
    
    to = get_timer("Shared")
    if length(size(original_bound.batch_Low)) > 3
        bound, img_size = convert_CROWN_Bound_batch(original_bound)
    else
        bound = original_bound
    end

    @timeit to "compute_bound"  l, u = compute_bound(bound) # reach_dim x batch, l is lower, u is upper

    mask_pos = l .>= 0 # reach_dim x batch
    mask_neg = u .<= 0 # reach_dim x batch
    mask_both = .!(mask_pos .| mask_neg) # reach_dim x batch
    lw = zeros(size(l)) # reach_dim x batch
    lb = zeros(size(l)) # reach_dim x batch
    uw = 0.25 .* ones(size(l)) # reach_dim x batch
    ub = ones(size(l)) # reach_dim x batch
    y_l = sigmoid_fast(l)
    y_u = sigmoid_fast(u)
    k = (y_u .- y_l) ./ (clamp.(u .- l, 1.0e-8, Inf))
    k_direct = k

    uw[mask_neg] = k_direct[mask_neg]
    b = y_l .- l .* k_direct
    ub[mask_neg] = b[mask_neg]

    lw[mask_pos] = k_direct[mask_pos]
    b = y_l .- l .* k_direct
    lb[mask_pos] = b[mask_pos]

    # TODO: pre-compute relaxation
    # similar to https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/auto_LiRPA/operators/tanh.py#L65C5-L65C8

    # Use the middle point slope as the lower/upper bound. Not optimized.
    m = (u .+ l) ./ 2
    y_m = sigmoid_fast(m)
    k = dsigmoid(m)
    
    # Lower bound is the middle point slope for the case input upper bound <= 0.
    # Note that the upper bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).    
    lw[mask_neg] = k[mask_neg]
    b = y_m .- m .* k
    lb[mask_neg] = b[mask_neg]

    # Upper bound is the middle point slope for the case input lower bound >= 0.
    # Note that the lower bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
    uw[mask_pos] = k[mask_pos]
    b = y_m .- m .* k
    ub[mask_pos] = b[mask_pos]

    # Now handle the case where input lower bound <=0 and upper bound >= 0.
    # Without pre-computed bounds, we only use the direct line as the lower bound, when this direct line does not intersect with f.
    # This is only valid when the slope at the input lower bound has a slope greater than the direct line.
    mask_direct = mask_both .& (k_direct .< dsigmoid(l))
    lw[mask_direct] = k_direct[mask_direct]
    b = y_l .- l .* k_direct
    lb[mask_direct] = b[mask_direct]

    # Do the same for the upper bound side when input lower bound <=0 and upper bound >= 0.
    mask_direct = mask_both .& (k_direct .< dsigmoid(u))
    uw[mask_direct] = k_direct[mask_direct]
    b = y_l .- l .* k_direct
    ub[mask_direct] = b[mask_direct]
    return lw, uw, lb, ub
end

function dsigmoid(x)
    return sigmoid_fast(x) .* (1 .- sigmoid_fast(x))
end
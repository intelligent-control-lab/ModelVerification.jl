
function forward(model, input, prop_method)
    # input: batch x ... x ...

    # bfs start from model.input_nodes
    
    forward_reach = input
    for layer in model.layers
        if isa(layer, Dense)
            forward_reach, info = forward_layer(layer, forward_reach, prop_method, info)
        elseif isa(layer, SkipConnection)
            forward_reach, info = forward(layer.layers, input, prop_method)
        end
    end

    return forward_reach, info

end


function backward(model, output, prop_method)
    # output: batch x ... x ...
    
    # bfs start from model.output_nodes
    return back_reach, info
end


function propagate(prop_method, model, input, output)
    if prop_method.type == :forward
        forward_reach, info = forward(model, input, prop_method)
        return check_forward(forward_reach, output), info
    elseif prop_method.type == :backward
        back_reach, info = backward(model, output, prop_method)
        return check_backward(back_reach, input, output), info
    elseif prop_method.type == :adversarial
        throw("unimplemented")
        # couterexample_result, info = pgd(model, input, output, prop_method)
        # return couterexample_result, info
    end
end

function split(split_method::split_method, input, info)
    return input1, input2
end

function branching(branch_method::Branch_method, prop_method::Prop_method, problem)
    branches = [problem.input]
    branch_cnt = 0
    while length(branches) > 0:
        branch_cnt > method.branching.max_iter && break
        branch_cnt += 1
        branch = popfirst(branches)
        result, info = propagate(method, problem.model, branch, output)
        result === :holds && continue
        if result === :violate
            continue
        push!(branches, split(branch, method.branching, info))
    end
end

function verify(method, problem)
    # preprocessing
    return branching(method.branching_method, method.prop_method, problem)
end
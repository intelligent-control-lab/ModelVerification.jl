include("/home/verification/ModelVerification.jl/src/operator/dense.jl")
include("/home/verification/ModelVerification.jl/src/operator/relu.jl")
include("/home/verification/ModelVerification.jl/src/operator/check.jl")

function forward(model, input, solver, info)
    # input: batch x ... x ...

    # bfs start from model.input_nodes
    forward_reach = input
    for layer in model.layers
        if isa(layer, Dense)
            forward_reach, info = forward_layer(layer, forward_reach, solver, info)
        elseif isa(layer, SkipConnection)
            forward_reach, info = forward(layer.layers, input, solver, info)
        end
    end

    return forward_reach, info

end

function backward(model, output, prop_method, info)
    # output: batch x ... x ...
    
    # bfs start from model.output_nodes
    return back_reach, info
end

function propagate(method, model, input, output, info)
    if method.prop_method == :forward
        forward_reach, info = forward(model, input, method.solver, info)
        return check_forward(forward_reach, output), info
    elseif prop_method == :backward
        back_reach, info = backward(model, output, method.solver, info)
        return check_backward(back_reach, input, output), info
    elseif prop_method == :adversarial
        throw("unimplemented")
        # couterexample_result, info = pgd(model, input, output, prop_method)
        # return couterexample_result, info
    end
end

#= function split(split_method, input, info)
    return input1, input2
end
 =#
function branching(method, problem)
    branches = [problem.input]
    branch_cnt = 0
    while length(branches) > 0
        branch_cnt > method.branching_method.max_iter && break
        branch_cnt += 1
        branch = popfirst(branches)
        result, info = propagate(method, problem.model, branch, output, info)
        result === :holds && continue
        if result === :violate
            continue
        end
        push!(branches, split(branch, method.branching, info))
    end
end

function verify(method, problem)
    # preprocessing
    return branching(method.branching_method, problem)
end
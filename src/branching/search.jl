@with_kw struct BFS <: SearchMethod
    max_iter::Int64
    batch_size::Int64 = 1
end

function search_branches(search_method::BFS, split_method, prop_method, problem)
    branches = [(problem.input, problem.output, nothing)]
    batch_input = []
    batch_output = []
    batch_info = []
    for iter in 1:search_method.max_iter # BFS with max iteration
        length(branches) == 0 && break
        input, output, info = popfirst!(branches)
        push!(batch_input, input)
        push!(batch_output, output)
        push!(batch_info, info)
        if length(batch_input) >= search_method.batch_size || length(branches) == 0
            #println(length(batch_input))
            batch_result, batch_info = propagate(prop_method, problem.model, batch_input, batch_output, batch_info)
            for i in eachindex(batch_input)
                batch_result[i].status == :holds && continue
                batch_result[i].status == :violated && return batch_result[i]
                # batch_result[i].status == :unknown
                sub_branches = split_branch(split_method, problem.model, batch_input[i], batch_output[i], batch_info[i])
                branches = [branches; sub_branches]
            end
            batch_input = []
            batch_info = []
        end
    end
    length(branches) == 0 && return BasicResult(:holds)
    return BasicResult(:unknown)
end
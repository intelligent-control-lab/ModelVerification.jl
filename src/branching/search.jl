@with_kw struct BFS <: SearchMethod
    max_iter::Int64
    batch_size::Int64 = 1
end

"""
    search_branches(search_method, split_method, prop_method, problem, model_info)

(Kai)
Essentially THE FUNCTION that does the heavy lifting...
Think of `prop_method` as the solver. Well, it is.
This returns the result.

"""
function search_branches(search_method::BFS, split_method, prop_method, problem, model_info)
    branches = [(problem.input, problem.output)]
    batch_input = []
    batch_output = []
    to = get_timer("Shared")
    @timeit to "test" current_time = 0
    for iter in 1:search_method.max_iter # BFS with max iteration
        length(branches) == 0 && break
        input, output = popfirst!(branches)
        # println(iter)
        # println(input)

        push!(batch_input, input)
        push!(batch_output, output)
        if length(batch_input) >= search_method.batch_size || length(branches) == 0
            
            # println(batch_input)
            @timeit to "prepare_method" batch_out_spec, batch_info = prepare_method(prop_method, batch_input, batch_output, model_info)
            @timeit to "propagate" batch_bound, batch_info = propagate(prop_method, model_info, batch_info)
            @timeit to "process_bound" batch_bound, batch_info = process_bound(prop_method, batch_bound, batch_out_spec, model_info, batch_info)
            @timeit to "check_inclusion" batch_result = check_inclusion(prop_method, problem.Flux_model, batch_input, batch_bound, batch_out_spec)
            # println("batch_bound")
            # println(batch_bound)
            for i in eachindex(batch_input)
                batch_result[i].status == :holds && continue
                batch_result[i].status == :violated && return batch_result[i]
                # batch_result[i].status == :unknown
                @timeit to "split_branch" sub_branches = split_branch(split_method, problem.Flux_model, batch_input[i], batch_output[i], model_info, batch_info)
                branches = [branches; sub_branches]
            end
            
            batch_input = []
            batch_output = []
        end
    end
    length(branches) == 0 && return BasicResult(:holds)
    return BasicResult(:unknown)
end

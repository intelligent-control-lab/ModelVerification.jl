"""
    BFS <: SearchMethod

Breadth-first Search (BFS) used to iteratively go through the branches in the 
branch bank.
    
## Fields
- `max_iter` (`Int64`): Maximum number of iterations to go through the branches 
    in the branch bank.
- `batch_size` (`Int64`): Size of the batch. Defaults to 1.
"""
@with_kw struct BFS <: SearchMethod
    max_iter::Int64
    batch_size::Int64 = 1
end

"""
    search_branches(search_method, split_method, prop_method, problem, model_info)

Searches through the branches in the branch bank, `branches`, until the branch 
bank is empty or time out. In each iteration (up to `search_method.max_iter`), 
a batch of unverified branches will be extracted from the branch bank. Then, the 
following is performed to verify the model:
 
1. `prepare_method` initializes the bound of the start node of the 
computational graph based on the geometric representation and corresponding 
solver.
2. `propagate` propagates the bound from the start node to the end node of the 
computational graph.
3. `process_bound` processes the bounds resulting from the propagation 
accordingly to the solver, `prop_method`. For example, for Ai2-based methods, 
`process_bound` simply returns the bounds from the `propagate` step. However, 
for Crown-based methods, `process_bound` post-processes the bounds.
4. `check_inclusion` decides whether the bound of the end node, the reachable 
set, satisfies the output specification or not. 
    1. If not, i.e., `:violated`, then the counterexample is returned and the 
    verification process terminates.
    2. If yes, i.e., `:holds`, then the current branch is verified and the 
    function starts Step 1 again for the next branch, if any.
    3. If unknown, i.e., `:unknown`, further refinement of the problem is 
    preformed using `split_branch`, which divides the current branch into 
    smaller pieces and puts them into the branch bank for further verification. 
    Such `:unknown` status results due to the overapproximation introduced in 
    the verification process.

If the branch bank is empty after going through `search_method.max_iter` number 
of verification procedures, the model is verified to be valid and returns 
`:holds`. If the branch bank is not empty, the function returns `:unknown`.

## Arguments
- `search_method` (`BFS`): Breadth-first Search method for iteratively going 
    through the branches.
- `split_method`: Method for splitting the branches when further refinement is 
    needed. This inclueds methods such as `Bisect` and `BaBSR`.
- `prop_method`: Propagation method used for the verification process. This is 
    one of the `solver` used to verify the given model.
- `problem`: Problem definition for model verification.
- `model_info`: Structure containing the information of the neural network to be 
    verified.

## Returns
- `BasicResult(:holds)` if all the reachable sets are within the corresponding 
    output specifications in the batch.
- `BasicResult(:unknown)` if the function failed to make a firm decision within 
    the given time. This is due to the overapproximation introduced in the 
    verification process.
- `CounterExampleResult(:violated, x)` if a reachable set is not within the 
    corresponding output specification and there is a counterexample found.
"""
function search_branches(search_method::BFS, split_method, prop_method, problem, model_info)
    branches = [(problem.input, problem.output)]
    
    # println(branches)
    
    # return BasicResult(:unknown)

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

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
advance_split(max_iter::Int, search_method::BFS, split_method, prop_method, 
              problem, model_info)

Performs the splitting of the branches in the branch bank, `branches`, for a 
`max_iter` number of times. This is used in the `search_branches` function as 
serves as the first step of the verification process: populating the `branches` 
bank with initial branches.

## Arguments
- `max_iter` (`Int`): Maximum number of iterations to split the input
    specification.
- `search_method` (`BFS`): Breadth-first Search method for iteratively going 
    through the branches.
- `split_method`: Method for splitting the branches for the initial 
    population of the branch bank. This inclueds methods such as `Bisect`.
- `prop_method`: Propagation method used for the verification process. This is 
    one of the solvers used to verify the given model.
- `problem`: Problem definition for model verification. Include the model and 
    input and output specifications.
- `model_info`: Structure containing the information of the neural network to be
    verified.

## Returns
- `branches`: Array of branches to be verified, split from the initial input 
    specification.
"""
function advance_split(max_iter::Int, search_method::BFS, split_method, prop_method, problem, model_info)
    branches = [Branch(problem.input, problem.output, nothing)]
    for iter in 1:max_iter # BFS with max iteration
        branch = popfirst!(branches)
        sub_branches = split_branch(split_method, problem.Flux_model, branch.input, branch.output, nothing, model_info, nothing)
        branches = [branches; sub_branches]
    end
    return branches
end

"""
    search_branches(search_method::BFS, split_method, prop_method, 
                    problem, model_info)

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
    one of the solvers used to verify the given model.
- `problem`: Problem definition for model verification.
- `model_info`: Structure containing the information of the neural network to be 
    verified.
- `collect_bound`(optional): Default is false, whether return the verified 
    bound.
- `pre_split`(optional): nothing, the number of split before any propagation. 
    This is particularly useful for large input set that could lead to memory 
    overflow.

## Returns
- `BasicResult(:holds)` if all the reachable sets are within the corresponding 
    output specifications in the batch.
- `BasicResult(:unknown)` if the function failed to make a firm decision within 
    the given time. This is due to the overapproximation introduced in the 
    verification process.
- `CounterExampleResult(:violated, x)` if a reachable set is not within the 
    corresponding output specification and there is a counterexample found.
"""
struct Branch
    input
    output
    inheritance::Union{Dict, Nothing}
end
Branch(input, output) = Branch(input, output, Dict())

function unpack_batch_branch(batch_branch)
    batch_input = [branch.input for branch in batch_branch]
    batch_output = [branch.output for branch in batch_branch]
    batch_inheritance = [branch.inheritance for branch in batch_branch]
    return batch_input, batch_output, batch_inheritance
end

function search_branches(search_method::BFS, split_method, prop_method, problem, model_info; collect_bound=false, pre_split=nothing, verbose=false)
    to = get_timer("Shared")
    branches = [Branch(problem.input, problem.output, nothing)]
    if !isnothing(pre_split)
        @timeit to "advance_split" branches = advance_split(pre_split, search_method, split_method, prop_method, problem, model_info)
    end 
    
    batch_branch = []
    @timeit to "test" current_time = 0
    verified_bound = []
    for iter in 1:search_method.max_iter # BFS with max iteration
        verbose && println("iter: ", iter, "   remaining branches: ", length(branches))
        length(branches) == 0 && break
        branch = popfirst!(branches)
        push!(batch_branch, branch) 
        
        if length(batch_branch) >= search_method.batch_size || length(branches) == 0
            
            batch_input, batch_output, batch_inheritance = unpack_batch_branch(batch_branch)
            @timeit to "prepare_method" batch_out_spec, batch_info = prepare_method(prop_method, batch_input, batch_output, batch_inheritance, model_info)
            
            # println(typeof(batch_output[1]))
            # println(typeof(batch_out_spec[1]))
            # println(batch_out_spec[1])
            # @assert false
            
            @timeit to "propagate" batch_bound, batch_info = propagate(prop_method, model_info, batch_info)
            @timeit to "process_bound" batch_bound, batch_info = process_bound(prop_method, batch_bound, batch_out_spec, model_info, batch_info)
            # println(typeof(batch_out_spec[1]))
            @timeit to "check_inclusion" batch_result = check_inclusion(prop_method, problem.Flux_model, batch_input, batch_bound, batch_out_spec)
            # println("batch_bound")
            # println(batch_bound)
            for i in eachindex(batch_input)
                batch_result[i].status == :holds && collect_bound && (push!(verified_bound, batch_bound[i]))
                batch_result[i].status == :holds && continue
                batch_result[i].status == :violated && return batch_result[i], verified_bound
                # batch_result[i].status == :unknown
                inheritance = get_inheritance(prop_method, batch_info, i, model_info)
                @timeit to "split_branch" sub_branches = split_branch(split_method, problem.Flux_model, batch_input[i], batch_output[i], inheritance, model_info, batch_info)
                branches = [branches; sub_branches]
            end
            batch_branch = []
        end
    end
    length(branches) == 0 && return BasicResult(:holds), verified_bound
    return BasicResult(:unknown), verified_bound
end

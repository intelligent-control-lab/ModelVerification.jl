__precompile__()
module ModelVerification

using JuMP
import Ipopt

# using GLPK, SCS # SCS only needed for Certify
# using PicoSAT # needed for Planet
using Polyhedra, CDDLib
using LazySets, LazySets.Approximations

using LinearAlgebra
using Parameters
using Interpolations # only for PiecewiseLinear

import LazySets: dim, HalfSpace, concretize # necessary to avoid conflict with Polyhedra
import Flux: flatten
using Requires

using Flux
using NNlib

using PaddedViews 
using Accessors

using Images, ImageIO
using ONNXNaiveNASflux, NaiveNASflux, .NaiveNASlib
using LinearAlgebra
using OpenCV
using Flux
using CUDA
using DataStructures
using Statistics
using Einsum
using Zygote

using ReachabilityAnalysis

using TimerOutputs

abstract type Solver end

"""
    SearchMethod

Algorithm for iterating through the branches, such as BFS (breadth-first search) 
and DFS (depth-first search). For an example, see the documentation on 
[`BFS`](@ref).
"""
abstract type SearchMethod end

"""
    SplitMethod

Algorithm to split an unknown branch into smaller pieces for further refinement.
Includes methods such as `Bisect` and `BaBSR`. For an example, see the 
documentation on [`Bisect`](@ref).
"""
abstract type SplitMethod end

"""
    PropMethod

Algorithm to propagate the bounds through the computational graph. This is the 
super-type for all the "solvers" used in the verification process, and is 
different from the functions defined in `propagate` folder. In other words, 
the `PropMethod` should be understood as the "solver" used in the verification 
process, while the functions in `propagate` folder are the "propagation" 
functions used in the verification process. The solvers include methods such as 
`Ai2` and `Crown`. For an example, see the documentation on [`Ai2`](@ref).
"""
abstract type PropMethod end

# @with_kw struct BranchMethod
#     search_method::SearchMethod
#     split_method::SplitMethod
# end
# abstract type BranchMethod end

# # For optimization methods:
# import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE
# JuMP.Model(solver::Solver) = Model(solver.optimizer)
# # define a `value` function that recurses so that value(vector) and
# # value(VecOfVec) works cleanly. This is only so the code looks nice.
# value(var::JuMP.AbstractJuMPScalar) = JuMP.value(var)
# value(vars::Vector) = value.(vars)
# value(val) = val

# include("utils/timer.jl")

include("spec/spec.jl")
export ImageConvexHull, InputSpec, OutputSpec
export get_linear_spec, get_image_linf_spec, classification_spec


include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")
include("utils/flux.jl")


export
    SearchMethod,
    SplitMethod,
    PropMethod,
    Problem,
    ODEProblem,
    BranchingMethod,
    Result,
    BasicResult,
    CounterExampleResult,
    AdversarialResult,
    ReachabilityResult,
    read_nnet,
    write_nnet,
    check_inclusion

# solve(m::Model; kwargs...) = JuMP.solve(m; kwargs...)
# export solve

# TODO: consider creating sub-modules for each of these.
include("solvers/solver.jl")
include("solvers/polytope.jl")
include("solvers/image-star.jl")
include("solvers/image-zono.jl")
include("solvers/crown.jl")
include("solvers/beta-crown.jl")
include("solvers/exact-reach.jl")
include("solvers/ode-taylor.jl")

include("propagate/propagate.jl")
include("propagate/propagate_neural_ode.jl")
include("propagate/operators/dense.jl")
include("propagate/operators/relu.jl")
include("propagate/operators/normalise.jl")
include("propagate/operators/stateless.jl")
include("propagate/operators/identity.jl")
include("propagate/operators/convolution.jl")
include("propagate/operators/bivariate.jl")
include("propagate/operators/util.jl")
include("propagate/operators/tanh.jl")
include("propagate/operators/sigmoid.jl")

include("attack/pgd.jl")


export Ai2, Ai2h, Ai2z, Box, ExactReach
export StarSet
export ImageStar, ImageZono
export Crown, AlphaCrown, BetaCrown
export ODETaylor

const tol = sqrt(eps())
const TOL = Ref(sqrt(eps()))
set_tolerance(x::Real) = (TOL[] = x)
export set_tolerance

include("branching/search.jl")
include("branching/split.jl")
include("branching/util.jl")

include("utils/preprocessing.jl")

include("utils/adv_input_bound.jl")

export BFS, Bisect, BFSBisect, BaBSR

macro timeout(seconds, expr, fail)
    println(seconds)
    quote
        tsk = @task $esc(expr)
        schedule(tsk)
        Timer($(esc(seconds))) do timer
            istaskdone(tsk) || Base.throwto(tsk, InterruptException())
        end
        try
            fetch(tsk)
        catch _
            $(esc(fail))
        end
    end
end

to = get_timer("Shared")
# verify(branch_method::BranchMethod, prop_method, problem) = search_branches(branch_method.search_method, branch_method.split_method, prop_method, problem)

"""
verify(search_method::SearchMethod, split_method::SplitMethod, 
       prop_method::PropMethod, problem::Problem; 
       time_out=86400, attack_restart=100, collect_bound=false, 
       summary=false, pre_split=nothing)

This is the main function for verification. It takes in a search method, 
a split method, a propagation method, and a problem, and returns a result. 
The result is of type `ResultInfo`, which is a wrapper for the following 
`Result` types: `BasicResult`, `CounterExampleResult`, `AdversarialResult`, 
`ReachabilityResult`, `EnumerationResult`, or timeout. 
For each `Result`, the `status` field is either `:violated`, `:verified`, or 
`:unknown`. Optional arguments can be passed to the function 
to control the timeout, the number of restarts for the attack, whether 
to collect the bounds for each branch, whether to print a summary of the 
verification process, and whether to pre-split the problem.

## Arguments
- `search_method` (`SearchMethod`): The search method, such as `BFS`, used to 
    search through the branches.
- `split_method` (`SplitMethod`): The split method, such as `Bisect`, used to 
    split the branches.
- `prop_method` (`PropMethod`): The propagation method, such as `Ai2`, used to 
    propagate the constraints.
- `problem` (`Problem`): The problem to be verified - consists of a network, 
    input set, and output set.
- `time_out` (`Int`): The timeout in seconds. Defaults to 86400 seconds, or 24 
    hours. If the timeout is reached, the function returns `:timeout`.
- `attack_restart` (`Int`): The number of restarts for the attack. Defaults to 100.
- `collect_bound` (`Bool`): Whether to collect the bounds for each branch.
- `summary` (`Bool`): Whether to print a summary of the verification process.
- `pre_split`: A function that takes in a `Problem` and returns a 
    `Problem` with the input set pre-split. Defaults to `nothing`.
- `search_adv_bound` (`Bool`): Whether to search the maximal input bound that can 
    pass the verification (get `:holds`) with the given setting.

## Returns
- The result is `ResultInfo`, the `status` field is either `:violated`, `:verified`, 
    `:unknown`, or `:timeout`. The info is a dictionary that contains other information.
"""
function verify(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem; 
                time_out=86400, 
                attack_restart=100, 
                collect_bound=false, 
                summary=false, 
                pre_split=nothing,
                search_adv_bound=false,
                verbose=false)
    to = get_timer("Shared")
    reset_timer!(to)
    # @timeit to "attack" res = attack(problem; restart=attack_restart)
    # (res.status == :violated) && (return res)
    @timeit to "prepare_problem" model_info, prepared_problem = prepare_problem(search_method, split_method, prop_method, problem)
    # println(time_out)   
    @timeit to "search_branches" res, verified_bounds = search_branches(search_method, split_method, prop_method, prepared_problem, model_info, collect_bound=collect_bound, pre_split=pre_split, verbose=verbose)
    # println(res.status)
    info = Dict()
    (res.status == :violated && res isa CounterExampleResult) && (info["counter_example"] = res.counter_example)
    collect_bound && (info["verified_bounds"] = verified_bounds)
    if res.status != :holds && search_adv_bound
        info["adv_input_scale"], info["adv_input_bound"] = search_adv_input_bound(search_method, split_method, prop_method, problem)# unknown or violated
    end
    summary && show(to) # to is TimerOutput(), used to profiling the code
    return ResultInfo(res.status, info)
end

export verify

include("utils/visualization.jl")
export visualize


end
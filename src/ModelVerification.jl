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

using TimerOutputs

abstract type Solver end

abstract type SearchMethod end
abstract type SplitMethod end

"""
    PropMethod

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

include("propagate/propagate.jl")
include("propagate/operators/dense.jl")
include("propagate/operators/relu.jl")
include("propagate/operators/normalise.jl")
include("propagate/operators/stateless.jl")
include("propagate/operators/identity.jl")
include("propagate/operators/convolution.jl")
include("propagate/operators/bivariate.jl")
include("propagate/operators/util.jl")

include("attack/pgd.jl")


export Ai2, Ai2h, Ai2z, Box, ExactReach
export StarSet
export ImageStar, ImageZono
export Crown, AlphaCrown, BetaCrown

const TOL = Ref(sqrt(eps()))
set_tolerance(x::Real) = (TOL[] = x)
export set_tolerance

include("branching/search.jl")
include("branching/split.jl")
include("branching/util.jl")

include("utils/preprocessing.jl")

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
function verify(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem; 
                time_out=86400, 
                attack_restart=100, 
                collect_bound=false, 
                summary=false, 
                pre_split=nothing,
                search_adv_bound=false)
    to = get_timer("Shared")
    reset_timer!(to)
    # @timeit to "attack" res = attack(problem; restart=attack_restart)
    # (res.status == :violated) && (return res)
    @timeit to "prepare_problem" model_info, prepared_problem = prepare_problem(search_method, split_method, prop_method, problem)
    # println(time_out)   
    @timeit to "search_branches" res, verified_bounds = search_branches(search_method, split_method, prop_method, prepared_problem, model_info, collect_bound=collect_bound, pre_split=pre_split)
    # println(res.status)
    info = Dict()
    (res.status == :violated && res isa CounterExampleResult) && (res["counter_example"] = res.counter_example)
    collect_bound && (info["verified_bounds"] = verified_bounds)
    (res.status != :holds && search_adv_bound) && (info["adv_input_bound"] = search_adv_input_bound(search_method, split_method, prop_method, problem))# unknown or violated
    summary && show(to) # to is TimerOutput(), used to profiling the code
    return ResultInfo(res.status, info)
end

function search_adv_input_bound(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem;
                        eps = 1e-3)
    l = 0
    r = 1
    while (r-l) > eps
        m = (l+r) / 2
        new_input = scale_set(problem.input, m)
        new_problem = Problem(problem.onnx_model_path, problem.Flux_model, new_input, problem.output)
        res = verify(search_method, split_method, prop_method, new_problem)
        if res.status == :holds
            l = m
            println("verified ratio: ",m)
        else
            println("falsified ratio: ",m)
            r = m
        end
    end
    return scale_set(problem.input, l)
end
function scale_set(set::Hyperrectangle, ratio)
    return Hyperrectangle(center(set), radius_hyperrectangle(set) * ratio)
end
function scale_set(set::ImageConvexHull, ratio)
    new_set = ImageConvexHull(copy(set.imgs))
    new_set.imgs[2:end] = [set.imgs[1] + (img - set.imgs[1]) * ratio for img in set.imgs[2:end]]
    return new_set
end


export verify

include("utils/visualization.jl")
export visualize


end
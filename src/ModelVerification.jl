module ModelVerification

# using JuMP

# using GLPK, SCS # SCS only needed for Certify
# using PicoSAT # needed for Planet
# using Polyhedra, CDDLib
using LazySets, LazySets.Approximations

using LinearAlgebra
using Parameters
using Interpolations # only for PiecewiseLinear

import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

using Requires

using Flux

abstract type Solver end

abstract type SearchMethod end
abstract type SplitMethod end
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

include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")

function __init__()
  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("utils/flux.jl")
end

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
include("propagate/propagate.jl")
include("propagate/check.jl")
include("propagate/solver.jl")
include("propagate/operators/dense.jl")
include("propagate/operators/relu.jl")
include("propagate/operators/identity.jl")
include("propagate/operators/util.jl")

# verify(branch_method::BranchMethod, prop_method, problem) = search_branches(branch_method.search_method, branch_method.split_method, prop_method, problem)
verify(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem) = 
    search_branches(search_method, split_method, prop_method, problem)
export verify

export Ai2, Ai2h, Ai2z, Box

const TOL = Ref(sqrt(eps()))
set_tolerance(x::Real) = (TOL[] = x)
export set_tolerance

include("branching/search.jl")
include("branching/split.jl")
include("branching/util.jl")

export BFS, Bisect, BFSBisect

end
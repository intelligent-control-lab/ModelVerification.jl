module ModelVerification

# using JuMP

# using GLPK, SCS # SCS only needed for Certify
# using PicoSAT # needed for Planet
# using Polyhedra, CDDLib
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


include("spec/spec.jl")

export ImageConvexHull, InputSpec, OutputSpec, classification_spec


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
include("propagate/solver.jl")
include("propagate/bound.jl")
include("propagate/propagate.jl")
include("propagate/check.jl")
include("propagate/operators/dense.jl")
include("propagate/operators/relu.jl")
include("propagate/operators/normalise.jl")
include("propagate/operators/stateless.jl")
include("propagate/operators/identity.jl")
include("propagate/operators/convolution.jl")
include("propagate/operators/bivariate.jl")
include("propagate/operators/util.jl")


export Ai2, Ai2h, Ai2z, Box
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

# verify(branch_method::BranchMethod, prop_method, problem) = search_branches(branch_method.search_method, branch_method.split_method, prop_method, problem)
function verify(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem)
    model_info, problem = prepare_problem(search_method, split_method, prop_method, problem)
    search_branches(search_method, split_method, prop_method, problem, model_info)
end

export verify

include("utils/visualization.jl")
export visualize


end
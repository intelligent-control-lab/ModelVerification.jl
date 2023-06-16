include("vnnlib_parser.jl")
using Interpolations
using Flux
using Polyhedra
using LazySets
using ONNX
import Umlaut: Tape, play!

@testset "ai2" begin
    onnx_file = "/home/verification/vnncomp2021/benchmarks/acasxu/ACASXU_run2a_1_1_batch_2000.onnx"
    vnnlib_file = "/home/verification/vnncomp2021/benchmarks/acasxu/prop_1.vnnlib"
    n_in = 5
    n_out = 5
    specs = read_vnnlib_simple(vnnlib_file, n_in, n_out)
    X_range, Y_cons = specs[1]
    lb = [bd[1] for bd in X_range]
    ub = [bd[2] for bd in X_range]
    X = Hyperrectangle(low = lb, high = ub)
    initial = rand(Float32, 5)
    flux_model = ONNX.load(onnx_file, initial)
    info = nothing
    branch = branching_method(:nothing, 100)
    solver = method(Ai2(), :forward, branch)
    holds, holds_info = propagate(solver, flux_model, in_hpoly, convert(HPolytope, out_superset), info)
    violated, holds_info = propagate(solver, flux_model, in_hpoly, convert(HPolytope, out_overlapping), info)
    @test holds.status    ∈ (:holds, :unknown)
    @test violated.status ∈ (:violated, :unknown)
#end
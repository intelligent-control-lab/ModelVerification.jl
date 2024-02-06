using NeuralVerification, LazySets, LinearAlgebra
using Test

#net_path = "/home/verification/ModelVerification.jl/test/networks/"

#include("/home/verification/ModelVerification.jl/src/operator/unit_test.jl")

onnx_file = "/home/verification/vnncomp2021/benchmarks/acasxu/ACASXU_run2a_1_1_batch_2000.onnx"

include("/home/verification/ModelVerification.jl/src/operator/test.jl")
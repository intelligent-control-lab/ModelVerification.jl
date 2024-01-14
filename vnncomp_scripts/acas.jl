using ModelVerification
using LazySets
using PyCall
using CSV
using Flux
using DataFrames
using ONNXNaiveNASflux 
include("ModelVerification.jl/vnncomp_scripts/vnnlib_parser.jl")

function onnx_to_nnet(onnx_file)
    # ACASXU onnx file considers constants as pseudo-input nodes with given parameters.
    # e.g. for the first Sub node, the inputAvg is given in the onnx file, but it is also considered an input node
    # The current onnx parser doesn't distinguish these pseudo-input from nominal input nodes
    # Therefore we use nnex to read onnx then convert it back to onnx to avoid this issue.
    # The nnet package will converted these pseudo-input nodes to constants.
    pushfirst!(PyVector(pyimport("sys")."path"), "./ModelVerification.jl")
    nnet = pyimport("NNet")
    use_gz = split(onnx_file, ".")[end] == "gz"
    if use_gz
        onnx_file = onnx_file[1:end-3]
    end
    nnet_file = onnx_file[1:end-4] * "nnet"
    isfile(nnet_file) && return
    nnet.onnx2nnet(onnx_file, nnetFile=nnet_file)
end

function verify_an_instance(onnx_file, spec_file, search_method, split_method, prop_method, timeout)
    use_gz = split(onnx_file, ".")[end] == "gz"
    onnx_to_nnet(onnx_file)
    nnet_file = use_gz ? onnx_file[1:end-7] * "nnet" : onnx_file[1:end-4] * "nnet"
    net = ModelVerification.read_nnet(nnet_file)
    ###### TODO: change this ad-hoc remedy for nnet read ######
    net.layers[1] = ModelVerification.Layer(net.layers[1].weights, net.layers[1].bias, ModelVerification.Id())
    flux_model = Flux.Chain(net)
    n_in = size(net.layers[1].weights)[2]
    n_out = length(net.layers[end].bias)
    specs = read_vnnlib_simple(spec_file, n_in, n_out)
    current_time = 0
    for spec in specs
        X_range, Y_cons = spec
        lb = [bd[1] for bd in X_range]
        ub = [bd[2] for bd in X_range]
        X = Hyperrectangle(low = lb, high = ub)
        res = nothing
        A = []
        b = []
        for Y_con in Y_cons
            A = hcat(Y_con[1]...)'
            b = Y_con[2]
            Yc = HPolytope(A, b)
            Y = Complement(Yc)
            problem = Problem(flux_model, X, Y)
            res = @timed verify(search_method, split_method, prop_method, problem)
            current_time += res.time 
            # println(current_time)
            if(current_time >= timeout)
                println("timed out at:", current_time)
                return "unknown"
            end
            res.value.status == :violated && (return "violated")
            res.value.status == :unknown && (return "unknown")
        end
    end
    return "holds"
end

function run_all(instance_csv, result_csv, search_method, split_method, prop_method)
    file = CSV.File(instance_csv, header=false)
    dirpath = dirname(instance_csv)
    df = DataFrame()
    for (index,row) in enumerate(file)
        println("Instance $index.")
        onnx_file = joinpath(dirpath, row[1])
        vnnlib_file = joinpath(dirpath, row[2])
        timeout = row[3]
        onnx_to_nnet(onnx_file)
        result = @timed verify_an_instance(onnx_file, vnnlib_file, search_method, split_method, prop_method, timeout)
        println(result)
        push!(df, result)
    end
    CSV.write(result_csv, df)
end

# expect violated
timeout = 116
onnx_file = "vnncomp2023_benchmarks/benchmarks/acasxu/onnx/ACASXU_run2a_1_2_batch_2000.onnx"
spec_file = "vnncomp2023_benchmarks/benchmarks/acasxu/vnnlib/prop_2.vnnlib"
result = @timed verify_an_instance(onnx_file, spec_file, search_method, split_method, prop_method, timeout)
using Revise
using LazySets
using ModelVerification
using PyCall
using CSV
using ONNX
using Flux
using DataFrames
using ONNXNaiveNASflux 
include("vnnlib_parser.jl")

function onnx_to_nnet(onnx_file)
    pushfirst!(PyVector(pyimport("sys")."path"), dirname(dirname(@__FILE__)))
    nnet = pyimport("NNet")
    use_gz = split(onnx_file, ".")[end] == "gz"
    if use_gz
        onnx_file = onnx_file[1:end-3]
    end
    nnet_file = onnx_file[1:end-4] * "nnet"
    isfile(nnet_file) && return
    nnet.onnx2nnet(onnx_file, nnetFile=nnet_file)
end

function verify_an_instance(onnx_file, spec_file, timeout)
    use_gz = split(onnx_file, ".")[end] == "gz"
    nnet_file = use_gz ? onnx_file[1:end-7] * "nnet" : onnx_file[1:end-4] * "nnet"
    net = ModelVerification.read_nnet(nnet_file)
    ###### TODO: change this ad-hoc remedy for nnet read ######
    net.layers[1] = ModelVerification.Layer(net.layers[1].weights, net.layers[1].bias, ModelVerification.Id())
    flux_model = Flux.Chain(net)
    new_onnx_file = "/home/verification/ModelVerification.jl/tmp/" * onnx_file[end-31:end]
    ONNXNaiveNASflux.save(new_onnx_file, flux_model, (5,5))
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
        # println("fuck")
        # println(Y_cons)
        for Y_con in Y_cons
            A = hcat(Y_con[1]...)'
            b = Y_con[2]
            Yc = HPolytope(A, b)
            Y = Complement(Yc)
            search_method = BFS(max_iter=10, batch_size=512)
            split_method = Bisect(1)
            prop_method = AlphaCrown(Crown(true, true), true, false, Flux.Optimiser(Flux.ADAM(0.1)), 10)#Crown(true, true)
            problem = Problem(new_onnx_file, X, Y)
            res = @timed verify(search_method, split_method, prop_method, problem)
            current_time += res.time 
            println(current_time)
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

function run_all(instance_csv, result_csv)
    file = CSV.File(instance_csv, header=false)
    dirpath = dirname(instance_csv)
    df = DataFrame()
    for (index,row) in enumerate(file)
        println("Instance $index.")
        onnx_file = joinpath(dirpath, row[1])
        vnnlib_file = joinpath(dirpath, row[2])
        timeout = row[3]
        onnx_to_nnet(onnx_file)
        result = @timed verify_an_instance(onnx_file, vnnlib_file, timeout)
        println(result)
        push!(df, result)
    end
    CSV.write(result_csv, df)
end

run_all("/home/verification/vnncomp2021/benchmarks/acasxu/acasxu_instances.csv", "./alphacrown_result.csv")
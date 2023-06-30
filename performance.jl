using Revise 
using LazySets
using ModelVerification
using PyCall
using CSV
using ONNX
using Flux
include("/home/verification/ModelVerification.jl/vnnlib_parser.jl")
function onnx_to_nnet(onnx_file)
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
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
    n_in = size(net.layers[1].weights)[2]
    n_out = length(net.layers[end].bias)
    flux_model = Flux.Chain(net)
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
            Y = HPolytope(-A, -b)
            search_method = BFS(max_iter=1e5, batch_size=4)
            split_method = Bisect(1)
            prop_method = Ai2z()
            problem = Problem(flux_model, X, Y)
            res = @timed verify(search_method, split_method, prop_method, problem)
            current_time += res.time 
            if(current_time >= timeout)
                print(111111111111)
                print("\n")
                print(current_time)
                print("\n")
                print(111111111)
                print("\n")
                return "unknown"
            end

            res.value.status == :violated && (return "violated")
            res.value.status == :unknown && (return "unknown")
        end
    end
    return "holds"
end

function main(args)
    file = CSV.File(args, header=false)
    dirpath = dirname(args) * "/"
    outpath = dirpath * "out_MV.txt"
    result_file = open("/home/verification/ModelVerification.jl/output_MV.txt", "w")
    all_time = 0
    ave_time = 0
    instance_num = -1
    max_time = 0
    min_time = Inf
    hold_number = 0
    max_memory = 0
    all_memory = 0
    ave_memory = 0
    for row in file
        instance_num += 1
        onnx_file = dirpath * row[1] 
        vnnlib_file = dirpath * row[2]
        timeout = row[3]
        onnx_to_nnet(onnx_file)
        result = @timed verify_an_instance(onnx_file, vnnlib_file, timeout)
        print(instance_num)
        print("\n")
        print(result)
        print("\n") 
        all_time += result.time
        all_memory += result.bytes 
        if result.time > max_time
            max_time = result.time
        elseif result.time < min_time
            min_time = result.time
        end

        if result.bytes > max_memory
            max_memory = result.bytes
        end

        if(result.value === "holds")
            hold_number += 1
        end

        text = "This is instance $instance_num.\n"
        write(result_file, text)
        write(result_file, result.value)
        write(result_file, "\n")
    end
    ave_time = all_time / (instance_num + 1)
    ave_memory = all_memory / (instance_num + 1)
    print(ave_time)
    print("\n")
    print(max_time)
    print("\n")
    print(min_time)
    print("\n")
    print(hold_number)
    print("\n")
    text = "average time is $ave_time.\n"
    write(result_file, text)
    write(result_file, "\n")
    text = "max time is $max_time.\n"
    write(result_file, text)
    write(result_file, "\n")
    text = "min time is $min_time.\n"
    write(result_file, text)
    write(result_file, "\n")
    text = "holds number is $hold_number.\n"
    write(result_file, text)
    write(result_file, "\n")
    text = "average bytes is $ave_memory.\n"
    write(result_file, text)
    write(result_file, "\n")
    text = "max bytes is $max_memory.\n"
    write(result_file, text)
    write(result_file, "\n")
    #close(file)
    close(result_file)
end

# main("/home/verification/vnncomp2021/benchmarks/acasxu/acasxu_instances.csv")
main("/home/verification/vnncomp2022_benchmarks/benchmarks/test/instances.csv")
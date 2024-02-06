include("/home/verification/ModelVerification.jl/src/operator/solver.jl")
include("/home/verification/ModelVerification.jl/src/main.jl")
using Interpolations
using Flux
using Polyhedra
using LazySets

function read_nnet(fname::String; last_layer_activation = Id())
    f = open(fname)
    line = readline(f)
    while occursin("//", line) #skip comments
        line = readline(f)
    end
    # number of layers
    nlayers = parse(Int64, split(line, ",")[1])
    # read in layer sizes
    layer_sizes = parse.(Int64, split(readline(f), ",")[1:nlayers+1])
    # read past additonal information
    for i in 1:5
        line = readline(f)
    end
    # i=1 corresponds to the input dimension, so it's ignored
    layers = Layer[read_layer(dim, f) for dim in layer_sizes[2:end-1]]
    push!(layers, read_layer(last(layer_sizes), f, last_layer_activation))
    
    return Network(layers), layers
end

function read_layer(output_dim::Int64, f::IOStream, act = ReLU())

    rowparse(splitrow) = parse.(Float64, splitrow[findall(!isempty, splitrow)]) 
     # first read in weights
     W_str_vec = [rowparse(split(readline(f), ",")) for i in 1:output_dim]
     weights = vcat(W_str_vec'...)
     # now read in bias
     bias_string = [split(readline(f), ",")[1] for j in 1:output_dim]
     bias = rowparse(bias_string)
     # activation function is set to ReLU as default
     return Layer(weights, bias, act)
end

function nnet_flux(model, layers)
    for i = 1:length(model)
        W, b = Flux.params(getindex(model,i))
        W .= layers[i].weights
        b .= layers[i].bias
    end
    return model
end

@testset "ai2" begin
    #small_nnet_file = "/home/verification/ModelVerification.jl/test/networks/small_nnet.nnet"
    small_nnet_file = net_path * "small_nnet.nnet"
    # small_nnet encodes the simple function 24*max(x + 1.5, 0) + 18.5
    small_nnet, layers = read_nnet(small_nnet_file, last_layer_activation = ReLU())
    model = Chain(
        Dense(1, 2, relu),
        Dense(2, 2, relu),
        Dense(2, 1, relu))
    flux_model = nnet_flux(model, layers)
    in_hyper  = Hyperrectangle(low = [-0.9], high = [0.9])
    in_hpoly  = convert(HPolytope, in_hyper)
    out_superset    = Hyperrectangle(low = [30.0], high = [80.0])    # 20.0 ≤ y ≤ 90.0
    out_overlapping = Hyperrectangle(low = [-1.0], high = [50.0])    # -1.0 ≤ y ≤ 50.0
    info = nothing
    branch = branching_method(:nothing, 100)    
    solver = method(Ai2(), :forward, branch)
    holds, holds_info = propagate(solver, flux_model, in_hpoly, convert(HPolytope, out_superset), info)
    violated, holds_info = propagate(solver, flux_model, in_hpoly, convert(HPolytope, out_overlapping), info)
    @test holds.status    ∈ (:holds, :unknown)
    @test violated.status ∈ (:violated, :unknown)
end
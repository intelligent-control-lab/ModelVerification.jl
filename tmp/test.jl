using LazySets
using ModelVerification
using PyCall
using CSV
using ONNX
using Flux
using Test
# using DataFrames
# import Flux: flatten
# using Flux: onehotbatch, onecold, flatten
# using Flux.Losses: logitcrossentropy
# using Statistics: mean
using CUDA
using MLDatasets: CIFAR10
using MLUtils: splitobs, DataLoader
using Accessors
using Profile
model = Chain([
    Flux.flatten,
    Dense(784, 200, relu),
    Dense(200, 10)
])
testmode!(model)
# image_seeds = CIFAR10(:train)[1:5].features # 32 x 32 x 3 x 5
image_seeds = [MNIST(:train)[i].features for i in 1:1]
# println(typeof(image_seeds[1][1,1,1,1]))
search_method = BFS(max_iter=1, batch_size=1)
split_method = Bisect(1)
output_set = BallInf(zeros(10), 1.0)
onnx_model_path = "/home/verification/ModelVerification.jl/debug.onnx"
Flux_model = model
image_shape = (28, 28, 1, 1)
#prop_method = ImageStar()
#@timed verify(search_method, split_method, prop_method, Problem(onnx_model_path, Flux_model, image_shape, image_seeds, output_set))
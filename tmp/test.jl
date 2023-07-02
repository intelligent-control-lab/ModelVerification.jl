using Revise
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
    Conv((3, 3), 3 => 8, relu, pad=SamePad(), stride=(2, 2)), #pad=SamePad() ensures size(output,d) == size(x,d) / stride.
    BatchNorm(8),
    MeanPool((2,2)),
    SkipConnection(
        Chain([
            Conv((5, 5), 8 => 8, relu, pad=SamePad(), stride=(1, 1))
            ]),
        +
    ),
    #ConvTranspose((3, 3), 8 => 4, relu, pad=SamePad(), stride=(2, 2)),#pad=SamePad() ensures size(output,d) == size(x,d) * stride.
    Flux.flatten,
    Dense(512, 100, relu),
    Dense(100, 10)
])
testmode!(model)
# image_seeds = CIFAR10(:train)[1:5].features # 32 x 32 x 3 x 5
image_seeds = [CIFAR10(:train)[i].features for i in 1:2]
# println(typeof(image_seeds[1][1,1,1,1]))
search_method = BFS(max_iter=1, batch_size=1)
split_method = Bisect(1)
output_set = BallInf(zeros(10), 1.0)
onnx_model_path = "/home/verification/ModelVerification.jl/mlp.onnx"
Flux_model = model
image_shape = (32, 32, 3, 5)
prop_method = ImageStar()
@timed verify(search_method, split_method, prop_method, Problem(onnx_model_path, Flux_model, image_shape, image_seeds, output_set))
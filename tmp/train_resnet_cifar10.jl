using Flux, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, @epochs, @treelike
using MLDatasets
#using CuArrays

include("dataloader.jl")

X, Y = CIFAR10.traindata();
tX, tY = CIFAR10.testdata();

X, Y = Float64.(X), onehotbatch(Float64.(Y), 0:9);
tX, tY = Float64.(tX), onehotbatch(Float64.(tY), 0:9);

trainloader = DataLoader(X, Y, batchsize=256, shuffle=true);
testloader = DataLoader(tX, tY, batchsize=256, shuffle=true);

## Flux does not support bias=false yet
conv3x3(ch::Pair{<:Int,<:Int}, stride::Int) = Conv((3, 3), ch, stride=stride, pad=1)

function align(ch::Pair{<:Int,<:Int}, stride::Int)
    if first(ch) != last(ch) || stride > 1
        Chain(conv3x3(ch, stride), BatchNorm(last(ch)))
    else
        identity
    end
end

struct ResBlock
    a
    f
end

ResBlock(ch::Pair{<:Int,<:Int}, stride::Int) = begin
    a = align(ch, stride)
    f = Chain(conv3x3(ch, stride),
               BatchNorm(last(ch)),
               x -> relu.(x),
               conv3x3(last(ch)=>last(ch), 1),
               BatchNorm(last(ch)))
    ResBlock(a, f)
end

@treelike ResBlock

(rb::ResBlock)(x) = relu.(rb.f(x) .+ rb.a(x))

function buildResBlocks(ch::Pair{<:Int,<:Int}, stride::Int, num_blocks::Int)
    blocks = [ResBlock(ch, stride)]
    for _ in 2:num_blocks
        push!(blocks, ResBlock(last(ch)=>last(ch), 1))
    end
    Chain(blocks...)
end

ResNet(num_classes::Int) =
    Chain(conv3x3(3=>16, 1),
          BatchNorm(16),
          x -> relu.(x),
          buildResBlocks(16=>16, 1, 2),
          buildResBlocks(16=>32, 2, 2),
          buildResBlocks(32=>64, 2, 2),
          MeanPool((8,8)),
          x -> reshape(x, :, size(x, 4)),
          Dense(64, num_classes))

m = ResNet(10) |> gpu

opt = ADAM(params(m))

loss(x, y) = logitcrossentropy(m(x |> gpu), y |> gpu)

@time(@epochs 1 Flux.train!(loss, trainloader, opt))

let n = 0
    for (x, y) in testloader
        n += sum(onecold(m(x |> gpu)) .== onecold(y |> gpu))
    end
    println("Accuracy: $(n/length(testloader))")
end
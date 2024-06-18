
@testset begin
    model = Chain([
        Conv((3, 3), 3 => 4, relu, pad=SamePad(), stride=(2, 2)), #pad=SamePad() ensures size(output,d) == size(x,d) / stride.
        BatchNorm(4),
        MeanPool((2,2)),
        SkipConnection(
            Chain([
                Conv((5, 5), 4 => 4, relu, pad=SamePad(), stride=(1, 1))
                ]),
            +
        ),
        ConvTranspose((3, 3), 4 => 2, relu, pad=SamePad(), stride=(2, 2)),#pad=SamePad() ensures size(output,d) == size(x,d) * stride.
        Flux.flatten,
        Dense(512, 100, relu),
        Dense(100, 10)
    ])
    model(rand(32,32,3,7))
# 16,16,4,7
# 16,16,4,7
# 8,8,4,7
# 8,8,4,7
# 8,8,4,7
# 16,16,2,7
# 512,7
# 100,7,
# 10,7
end
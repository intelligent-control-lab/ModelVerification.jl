```@meta
CurrentModule = ModelVerification
```

```@contents
Pages = ["network.md"]
Depth = 3
```

# Network

## Model
```@docs
Model
```

## Network
```@docs
Network
Layer{F<:ActivationFunction, N<:Number}
```

## Activation Functions
```@autodocs
Modules=[ModelVerification]
Pages=["activation.jl"]
```

## Helper Functions
```@docs
onnx_parse(onnx_model_path)
get_act(l)
n_nodes(L::Layer)
read_nnet(fname::String; last_layer_activation = Id())
read_layer(output_dim::Int64, f::IOStream, act = ReLU())
to_comment(txt)
print_layer(file::IOStream, layer)
print_header(file::IOStream, network; header_text="")
write_nnet(filename, network; header_text)
```
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
Below are the helper functions regarding [network loading (from file) & dumping (to file)](#network-loading-and-dumping), [property-related](#network-properties), [activation function operations](#activation-function-operations), [gradient-related operations](#gradient-operations), and [bound & specification related operations](#bound--specification-operations).

### Network loading and dumping
```@docs
onnx_parse(onnx_model_path)
read_nnet(fname::String; last_layer_activation = Id())
read_layer(output_dim::Int64, f::IOStream, act = ReLU())
to_comment(txt)
print_layer(file::IOStream, layer)
print_header(file::IOStream, network; header_text="")
write_nnet(filename, network; header_text)
build_flux_model(onnx_model_path)
get_chain(vertex)
purify_flux_model(model::Chain)
remove_flux_start_flatten(model::Chain)
build_onnx_model(path, model::Chain, input::InputSpec)
```

### Network properties
```@docs
get_act(l)
n_nodes(L::Layer)
get_sub_model(model_info, end_node)
compute_output(nnet::Network, input)
```

### Activation function operations
```@docs
get_activation(L::Layer{ReLU}, x::Vector)
get_activation(L::Layer{Id}, args...)
get_activation(nnet::Network, x::Vector{Float64})
get_activation(nnet::Network, input::Hyperrectangle)
get_activation(nnet::Network, bounds::Vector{Hyperrectangle})
get_activation(L::Layer{ReLU}, bounds::Hyperrectangle)
approximate_act_map(act::ActivationFunction, input::Hyperrectangle)
approximate_act_map(layer::Layer, input::Hyperrectangle)
```

### Gradient operations
```@docs
get_gradient(nnet::Network, x::Vector)
act_gradient(act::ReLU, z_hat::Vector)
act_gradient(act::Id, z_hat::Vector)
relaxed_relu_gradient(l::Real, u::Real)
act_gradient_bounds(nnet::Network, input::AbstractPolytope)
get_gradient_bounds(nnet::Network, input::AbstractPolytope)
get_gradient_bounds(nnet::Network, LΛ::Vector{<:AbstractVector}, UΛ::Vector{<:AbstractVector})
```

### Bound & Specification operations
```@docs
interval_map(W::Matrix, l::AbstractVecOrMat, u::AbstractVecOrMat)
get_bounds(nnet::Network, input; before_act::Bool = false)
get_bounds(problem::Problem; kwargs...)
isbounded(input)
is_hypercube(set::Hyperrectangle)
is_halfspace_equivalent(set)
UnboundedInputError
```
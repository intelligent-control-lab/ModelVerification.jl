"""
    AbstractNetwork

Abstrct type representing networks. Note that "model", "network", "neural 
network", "deep neural network" are used interchangeably throughout the toolbox.
"""

abstract type AbstractNetwork end

"""
    Layer{F, N}

Consists of `weights` and `bias` for linear mapping, and `activation` for 
nonlinear mapping.

## Fields
 - `weights::Matrix{N}`
 - `bias::Vector{N}`
 - `activation::F`

See also: [`Network`](@ref)
"""
struct Layer{F<:ActivationFunction, N<:Number}
    weights::Matrix{N}
    bias::Vector{N}
    activation::F
end

"""
    Network([layer1, layer2, layer3, ...])

A vector of layers.

## Fields
- `layers` (`Vector{Layer}`): Layers of the network, including the output layer.

See also: [`Layer`](@ref)
"""
struct Network <: AbstractNetwork
    layers::Vector{Layer} # layers includes output layer
end

"""
    n_nodes(L::Layer)

Returns the number of neurons in a layer.
"""
n_nodes(L::Layer) = length(L.bias)


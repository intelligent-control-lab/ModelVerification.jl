# Helper Functions

## Flux-to-Network, Network-to-Flux
```Julia
network(c::Chain) = Network([layer.(c.layers)...])
```
Converts `Flux.Chain` to a `Network`.

```Julia
Flux.Chain(m::Network) = _flux(m)
```
Converts `Network` to a `Flux.Chain`.

## Testing Functions
```@autodocs
Modules=[ModelVerification]
Pages=["testing_utils.jl"]
```
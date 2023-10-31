# Helper Functions
[](TODO: can rename helper to whatever makes the most sense)

[](```@contents)
[](    Pages = ["functions.md"])
[](    Depth = 3)
[](```)

[](TODO: Since most of these function are not exported they have to be called with ModelVerification.[])
[](Should consider whether we want to list unexported functions online at all.)


```@docs
ModelVerification.read_nnet
ModelVerification.init_layer
ModelVerification.compute_output
ModelVerification.get_activation
ModelVerification.get_gradient
ModelVerification.act_gradient
ModelVerification.act_gradient_bounds
ModelVerification.interval_map
ModelVerification.get_bounds
ModelVerification.linear_transformation
ModelVerification.split_interval
```
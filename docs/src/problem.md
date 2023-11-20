```@contents
Pages=["problem.md"]
Depth = 3
```

# Problem Outline

Verification checks if the input-output relationships of a function, specifically deep neural networks in this case, hold. For an input specification imposed by a set $\mathcal{X}\subseteq \mathcal{D}_x$, we would like to check if the corresponding output of the function is contained in an output specification imposed by a set $\mathcal{Y}\subseteq \mathcal{D}_y$:

$$x\in\mathcal{X} \Longrightarrow y = f(x) \in \mathcal{Y}.$$



```@autodocs
Modules=[ModelVerification]
Pages=["problem.jl"]
```

## Network
_Details on [Network](./network.md)_

## Safety Property
_Details on [Input-Output Specification](./safety_spec.md)_


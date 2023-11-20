```@contents
Pages=["safety_spec.md"]
Depth=3
```

# Safety Specifications

## Input-Output Specification
Verification checks if the input-output relationships of a function, specifically deep neural networks in this case, hold. For an input specification imposed by a set $\mathcal{X}\subseteq \mathcal{D}_x$, we would like to check if the corresponding output of the function is contained in an output specification imposed by a set $\mathcal{Y}\subseteq \mathcal{D}_y$:

$$x\in\mathcal{X} \Longrightarrow y = f(x) \in \mathcal{Y}.$$


## Safety Property
A safety property is essentially an input-output relationship for the model we want to verify. In general, the constraints for the input set $\mathcal{X}$ and the output set $\mathcal{Y}$ can have any geometry. For the sake of simplicity, [ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) uses convex polytopes and the complement of a polytope to encode the input and output specifications. Specifically, our implementation utilizes the geometric definitions of [LazySets](https://juliareach.github.io/LazySets.jl/dev/), a Julia package for calculus with convex sets. The following section dives into the geometric representations [ModelVerification.jl](https://github.com/intelligent-control-lab/ModelVerification.jl) uses and the representations required for each solver. 

## Geometric Representation
Different solvers implemented in ModelVerification.jl require the input-output specification formulated with particular geometries. We report here a brief overview of the sets we use. For specifics, please read [_Algorithms for Verifying Deep Neural Networks_ by C. Liu, et al.](https://arxiv.org/abs/1903.06758)  and [Sets in `LazySets.jl`](https://juliareach.github.io/LazySets.jl/dev/lib/interfaces/#Set-Interfaces).

- HR = `Hyperrectangle`
- HS = `HalfSpace`
- HP = `HPolytope`
- ZT = `Zonotope`
- SS = `StarSet`
- IS = `ImageStar`
- PC = `PolytopeComplement`

| **Solver**             | **Input Set**  | **Output Set** |
| ---------------------- | -------------- | -------------- |
| Ai2                    | HR, ZT, HP, SS | HP (bounded)   |
| CROWN                  | HR             | HP (bounded)   |
| $\alpha$-CROWN         | HR             | HP (bounded)   |
| $\beta$-CROWN          | HR             | HP (bounded)   |
| $\alpha$-$\beta$-CROWN | HR             | HP (bounded)   |


### Hyperrectangle ([`Hyperrectangle`](https://juliareach.github.io/LazySets.jl/dev/lib/sets/Hyperrectangle/#def_Hyperrectangle))
Corresponds to a high-dimensional rectangle, defined by

$$|x-c| \le r,$$

where $c\in\mathbb{R}^{k_0}$ is the center of the hyperrectangle and $r\in\mathbb{R}^{k_0}$ is the radius of the hyperrectangle.

### HalfSpace ([`HalfSpace`](https://juliareach.github.io/LazySets.jl/dev/lib/sets/HalfSpace/))
Represented by a single linear inequality constraint

$$c^\top x \le d,$$

where $c\in\mathbb{R}^{k_0}$ and $d\in\mathbb{R}$.

### Halfspace-Polytope ([`HPolytope`](https://juliareach.github.io/LazySets.jl/dev/lib/sets/HPolytope/#def_HPolytope))
[`HPolytope`](https://juliareach.github.io/LazySets.jl/dev/lib/sets/HPolytope/#def_HPolytope) uses a set of linear inequality constraints to represent a convex polytope, i.e., it is a bounded set defined using an intersection of half-spaces.

$$Cx \le d,$$

where $C\in\mathbb{R}^{k\times k_0}, d\in\mathbb{R}^k$ with $k$ representing the number of inequality constraints.

### Zonotope

### StarSet

### ImageStar

### PolytopeComplement

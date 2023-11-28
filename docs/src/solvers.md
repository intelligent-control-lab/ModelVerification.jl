```@meta
CurrentModule = ModelVerification
```

```@contents
Pages=["solvers.md"]
Depth = 3
```
# Solvers

For most of the functions below, each solver has a unique dispatch defined.

## Variations of propagation methods
All the solvers are based on one of the following propagation methods.
```@docs
ForwardProp
BackwardProp
SequentialForwardProp
SequentialBackwardProp
BatchForwardProp
BatchBackwardProp
```

## Bound types
The bounds are based on the following abstract type `Bound`.
```@docs
Bound
```

## Preprocessing for the solver
`prepare_method` is the first step called in [`search_branches`](@ref search_branches(search_method::BFS, split_method, prop_method, problem, model_info)). It initializes the bounds of the start node of the computational graph based on the given branch and the geometric representation used by the solver, which is specified with the `prop_method`. For each solver, there is a unique `prepare_method` defined. For more information, refer to the documentation for each solver.
```@docs
prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
```

The following functions are used to retrieve information regarding each node in the model.
```@docs
init_propagation(prop_method::ForwardProp, batch_input, batch_output, model_info)
init_propagation(prop_method::BackwardProp, batch_input, batch_output, model_info)
```

The following functions are used to either retrieve or process the safety specification.
```@docs
init_batch_bound(prop_method::ForwardProp, batch_input, batch_output)
init_batch_bound(prop_method::BackwardProp, batch_input, batch_output)
init_bound(prop_method::ForwardProp, input)
init_bound(prop_method::BackwardProp, output)
process_bound
```

## Checking inclusion

```@docs
check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, batch_reach::AbstractArray, batch_output::AbstractArray)
```

# Ai2
```@docs
Ai2
StarSet
prepare_method(prop_method::StarSet, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
compute_bound(bound::Zonotope)
compute_bound(bound::Star)
init_bound(prop_method::StarSet, input::Hyperrectangle) 
check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::LazySet)
check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::Complement)
```

# ImageStar
```@docs
ImageStar
ImageStarBound
prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageStar, problem::Problem)
prepare_method(prop_method::ImageStar, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
init_bound(prop_method::ImageStar, ch::ImageConvexHull) 
assert_zono_star(bound::ImageStarBound)
compute_bound(bound::ImageStarBound)
center(bound::ImageStarBound)
check_inclusion(prop_method::ImageStar, model, input::ImageStarBound, reach::LazySet, output::LazySet)
```

# ImageZono
```@docs
ImageZono
ImageZonoBound
prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageZono, problem::Problem)
init_bound(prop_method::ImageZono, ch::ImageConvexHull) 
init_bound(prop_method::ImageZono, bound::ImageStarBound)
compute_bound(bound::ImageZonoBound)
center(bound::ImageZonoBound)
check_inclusion(prop_method::ImageZono, model, input::ImageZonoBound, reach::LazySet, output::LazySet)
```

# Crown

# $\alpha$-Crown

# $\beta$-Crown
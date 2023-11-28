```@meta
CurrentModule = ModelVerification
```

```@contents
Pages=["branching.md"]
Depth = 3
```

# Branching
The "branch" part of the _Branch and Bound_ paradigm for verification algorithms. The `branching` folder contains algorithms for dividing the input set into searchable smaller sets, which we call "branches." 

The `search.jl` module includes algorithms to iterate over all branches, such as BFS (Breadth-first Search) and DFS (Depth-first Search). The `search.jl\search_branches` function is of particular importance since it executes the verification procedure.

The `split.jl` module includes algorithms to split an unknown branch, such as bisect, sensitivity analysis, etc. The `split.jl\split_branch` function divides the unknown branches into smaller pieces and put them back to the branch bank for future verification. This is done so that we can get a more concrete answer by refining the problem in case the over-approximation introduced in the verification process prevents us from getting a firm result.

## Search
```@docs
BFS
search_branches(search_method::BFS, split_method, prop_method, problem, model_info)
```

## Split

### Bisection
```@docs
Bisect
split_branch(split_method::Bisect, model::Chain, input::Hyperrectangle, output, model_info, batch_info)
split_branch(split_method::Bisect, model::Chain, input::LazySet, output, model_info, batch_info)
split_branch(split_method::Bisect, model::Chain, input::ImageStarBound, output)
split_branch(split_method::Bisect, model::Chain, input::ImageZonoBound, output)
split_branch(split_method::Bisect, model::Chain, input::ImageStarBound, output, model_info, batch_info)
split_branch(split_method::Bisect, model::Chain, input::ImageZonoBound, output, model_info, batch_info)
split_interval(dom::Hyperrectangle, i::Int64)
```

### Branch-and-bound
```@docs
BaBSR
split_branch(split_method::BaBSR, model::Chain, input::Tuple, output, model_info, batch_info)
split_beta(S_dict, score, split_relu_node, i, split_neurons_index_in_node, j, input, output)
vecsign_convert_to_original_size(index, vector, original)
vecmask_convert_to_original_size(index, original)
branching_scores_kfsb(model_info, batch_info, input)
topk(score, k, model_info)
```

### Input Gradient Split
```@docs
InputGradSplit
```
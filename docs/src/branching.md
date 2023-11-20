```@contents
Pages=["branching.md"]
Depth = 3
```

# Branching
The "branch" part of the _Branch and Bound_ paradigm for verification algorithms. The `branching` folder contains algorithms for dividing the input set into searchable smaller sets, which we call "branches." 

The `search.jl` module includes algorithms to iterate over all branches, such as BFS (Breadth-first Search) and DFS (Depth-first Search). The `search.jl\search_branches` function is of particular importance since it executes the verification procedure.

The `split.jl` module includes algorithms to split an unknown branch, such as bisect, sensitivity analysis, etc. The `split.jl\split_branch` function divides the unknown branches into smaller pieces and put them back to the branch bank for future verification. This is done so that we can get a more concrete answer by refining the problem in case the over-approximation introduced in the verification process prevents us from getting a firm result.

## Search
```@autodocs
Modules=[ModelVerification]
Pages=["search.jl"]
```

## Split
```@autodocs
Modules=[ModelVerification]
Pages=["split.jl"]
```
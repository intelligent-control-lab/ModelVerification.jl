
# Changes

## Already Implemented

In order to update to a newer Julia versions, I made the following changes:
- remove old ONNX reader (main reason for restricting newer versions) and replace it with `VNNLib.jl`
    - `VNNLib.jl` also has an ONNX reader, but it can **only read** ONNX, not write it. It also doesn't really convert the ONNX models to Flux (some ONNX operations just require some more manipulation), but transforms each node to a respective Julia type that can be executed using Flux *and additional Julia functions*.
- don't call `import Flux: flatten` because it collides with the same function from `ReachabilityAnalysis`, just use the prefixed version `Flux.flatten` instead
- add `src/LazySetsAdapter/RotatedHyperrectangle.jl` because `RotatedHyperrectangle` was removed from `LazySets` in version `3.0.0`.
- removed the constructor `BetaCrown(nothing)` because it was preventing precompilation.
    - the constructor `BetaCrown(nothing; use_gpu=false)` is still there (thus the constructor above should not have been necessary in the first place)
- change from `Flux.@functor` to `Flux.@layer` as recommended for Flux `0.15` and higher

Minor changes
- fixed introductory example in `README.md`
- added crude additional documentation to `ModelGraph` in `src/utils/preprocessing.jl`


## Planned to Implement

(Basically everything in `develop/onnx_parser_adaptions.jl`)

- transform models in `VNNLib.OnnxNet` intermediate representation to `Flux` code
    - Only necessary if you really need the Flux representation. (Why do you need it?)
- Convert `VNNlib.OnnxNet` to `ModelVerification.ModelGraph` 
    - this is only for convenience as `ModelGraph` is used all throughout `ModelVerification`, the fields almost have a one-to-one mapping
- Do this conversion in `prepare_problem`
- dispatch `propagate_layer_batch()` not on the `Flux` operation, but on the subtype of `VNNLib.OnnxParser.Node`
    - Many times you can just reuse the functions dispatching on the `Flux` operations. But some ONNX operations don't have a single equivalent `Flux` layer (e.g. if `TRANSPOSE` was set in `MatMul` or `Gemm`).



# Questions

- why are there separate ONNX to Flux conversions in `src/utils/problem.jl` and `src/utils/preprocessing.jl`?
    - Do you want to be able to just directly verify models trained in Julia as Flux models? (If that's the case, we could transform them into the intermediate representation used by `VNNLib.OnnxParser`)
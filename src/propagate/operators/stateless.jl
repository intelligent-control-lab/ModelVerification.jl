"""
    propagate_linear(prop_method, layer::typeof(flatten), 
                     bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a flatten layer. I.e., it flattens 
the `ImageStarBound` into a `Star` type.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(flatten)`): The layer operation to be used for propagation.
- `bound` (`ImageStarBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The flattened bound of the output layer represetned in `Star` type.
"""
propagate_linear(prop_method, layer::typeof(flatten), bound::ImageStarBound, batch_info) = 
    Star(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)), HPolyhedron(bound.A, bound.b))

"""
    propagate_linear(prop_method, layer::typeof(flatten), 
                     bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a flatten layer. I.e., it flattens 
the `ImageZonoBound` into a `Zonotope` type.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`typeof(flatten)`): The layer operation to be used for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The flattened bound of the output layer represetned in `Zonotope` type.
"""
propagate_linear(prop_method, layer::typeof(flatten), bound::ImageZonoBound, batch_info) =
    Zonotope(reshape(bound.center, :), reshape(bound.generators, :, size(bound.generators,4)))

"""
    propagate_linear(prop_method, layer::MeanPool, 
                     bound::ImageStarBound, batch_info)

Propagate the `ImageStarBound` bound through a mean pool layer. I.e., it applies
the mean pool operation to the `ImageStarBound` bound. The resulting bound is 
also of type `ImageStarBound`.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`MeanPool`): The mean pool operation to be used for propagation.
- `bound` (`ImageStarBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The mean pooled bound of the output layer represetned in `ImageStarBound` 
    type.
"""
function propagate_linear(prop_method, layer::MeanPool, bound::ImageStarBound, batch_info)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageStarBound(new_center, new_generators, bound.A, bound.b)
end

"""
    propagate_linear(prop_method, layer::MeanPool, 
                     bound::ImageZonoBound, batch_info)

Propagate the `ImageZonoBound` bound through a mean pool layer. I.e., it applies
the mean pool operation to the `ImageZonoBound` bound. The resulting bound is 
also of type `ImageZonoBound`.

## Arguments
- `prop_method`: The propagation method used for the verification problem.
- `layer` (`MeanPool`): The mean pool operation to be used for propagation.
- `bound` (`ImageZonoBound`): The bound of the input node.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- The mean pooled bound of the output layer represetned in `ImageZonoBound` 
    type.
"""
function propagate_linear(prop_method, layer::MeanPool, bound::ImageZonoBound, batch_info)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageZonoBound(new_center, new_generators)
end
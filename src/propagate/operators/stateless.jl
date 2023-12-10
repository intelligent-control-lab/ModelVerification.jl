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
- The flattened bound of the output layer represented in `Star` type.
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
- The flattened bound of the output layer represented in `Zonotope` type.
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
- The mean pooled bound of the output layer represented in `ImageStarBound` 
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
- The mean pooled bound of the output layer represented in `ImageZonoBound` 
    type.
"""
function propagate_linear(prop_method, layer::MeanPool, bound::ImageZonoBound, batch_info)
    new_center = layer(bound.center)
    new_generators = layer(bound.generators)
    return ImageZonoBound(new_center, new_generators)
end

function propagate_linear_batch(prop_method::Crown, layer::MeanPool, bound::CrownBound, batch_info)
    @assert length(size(bound.batch_Low)) > 3
    img_size = size(bound.batch_Low)[1:3]
    l, u = compute_bound(bound)
    img_low = reshape(l, (img_size..., size(l)[2]))
    img_up = reshape(u, (img_size..., size(u)[2]))
    new_low = layer(img_low)
    new_up = layer(img_up)
    batch_input = [ImageConvexHull([new_low[:,:,:,i], new_up[:,:,:,i]]) for i in size(new_low)[end]]
    new_crown_bound = init_batch_bound(prop_method, batch_input,nothing)
    return new_crown_bound
end

function propagate_linear_batch(prop_method::Crown, layer::typeof(Flux.flatten), bound::CrownBound, batch_info)
    bound, _ = convert_CROWN_Bound_batch(prop_method,bound)
    return bound
end


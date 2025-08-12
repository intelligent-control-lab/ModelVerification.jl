
"""
This file contains code to allow ModelVerification to work with networks parsed using VNNLib.ONNXParser.

The file currently only provides an adapter with functions dispatching on the type of the ONNXParser node and 
then directly calling the ModelVerification's code for the corresponding Flux layer.

For each node type `<OXP.NodeType>` defined in `VNNLib.OnnxParser`, we need to define the method:
- `_propagate_layer_batch(prop_method, node::<OXP.NodeType>, bound, batch_info)` (with an underscore prefix) This method redirects to the  already defined methods
for Flux layers.

To prevent ambiguities, we also need to define the following methods, that just redirect to the previous method (with the underscore prefix):
- `propagate_layer_batch(prop_method, node::<OXP.NodeType>, bound, batch_info)` (without an underscore prefix) 
- `propagate_layer_batch(prop_method::ForwardProp, node::<OXP.NodeType>, bound::AbstractArray, batch_info)` (without an underscore prefix)
"""


function _propagate_layer_batch(prop_method, node::OXP.DummyInputNode, bound, batch_info)
    propagate_layer_batch(prop_method, Flux.identity, bound, batch_info)
end

# need to define these methods to avoid ambiguities
propagate_layer_batch(prop_method, node::OXP.DummyInputNode, bound, batch_info) = _propagate_layer_batch(prop_method, node, bound, batch_info)
propagate_layer_batch(prop_method::ForwardProp, node::OXP.DummyInputNode, bound::AbstractArray, batch_info) = _propagate_layer_batch(prop_method, node, bound, batch_info)

function _propagate_layer_batch(prop_method, node::OXP.ONNXLinear, bound, batch_info)
    # we don't currently support transpose, because we directly call the code for the Flux dense layer 
    # provided in ModelVerification.
    # To handle transpose, we would need to also call code for transposing!
    @assert node.transpose == false "Transpose is currently not supported for ONNXLinear!"
    propagate_layer_batch(prop_method, node.dense, bound, batch_info)
end

propagate_layer_batch(prop_method, node::OXP.ONNXLinear, bound, batch_info) = _propagate_layer_batch(prop_method, node, bound, batch_info)
propagate_layer_batch(prop_method::ForwardProp, node::OXP.ONNXLinear, bound::AbstractArray, batch_info) = _propagate_layer_batch(prop_method, node, bound, batch_info)


function _propagate_layer_batch(prop_method, node::OXP.ONNXRelu, bound, batch_info)
    propagate_layer_batch(prop_method, Flux.relu, bound, batch_info)
end

propagate_layer_batch(prop_method, node::OXP.ONNXRelu, bound, batch_info) = _propagate_layer_batch(prop_method, node, bound, batch_info)
propagate_layer_batch(prop_method::ForwardProp, node::OXP.ONNXRelu, bound::AbstractArray, batch_info) = _propagate_layer_batch(prop_method, node, bound, batch_info)



# TODO: only keep these function until they are included in VNNLib.OnnxParser 


function Flux.outputsize(m::OXP.Node, inputsize::Tuple; padbatch=false)
    f = onnx_node_to_flux_layer(m)
    Flux.outputsize(f, inputsize; padbatch=padbatch)
end


OXP.onnx_node_to_flux_layer(node::OXP.DummyInputNode) = identity


function add_dummy_input_node(model::OnnxNet)
    @assert length(model.start_nodes) == 1 "Currently only one start node is supported."
    start_node_name = model.start_nodes[1]
    start_node = deepcopy(model.nodes[start_node_name])

    input_names = model.nodes[start_node.name].inputs
    # TODO: no guarantee that names are unique!
    output_names = [string(iname, "_$i") for (i, iname) in enumerate(input_names)]
    dummy_input = OXP.DummyInputNode(input_names, output_names, "DummyInput_$(Int(ceil(rand()*1000)))")

    start_node.inputs .= output_names

    start_nodes = [dummy_input.name]
    final_nodes = model.final_nodes

    nodes = deepcopy(model.nodes)
    # need to update the start node, because we copied it
    nodes[start_node.name] = start_node
    nodes[dummy_input.name] = dummy_input

    output_dict = deepcopy(model.output_dict)
    for oname in output_names
        output_dict[oname] = dummy_input.name
    end

    node_prevs = deepcopy(model.node_prevs)
    node_prevs[dummy_input.name] = []
    node_prevs[start_node.name] = [dummy_input.name]

    node_nexts = deepcopy(model.node_nexts)
    node_nexts[dummy_input.name] = [start_node.name]

    input_shapes = deepcopy(model.input_shapes)
    output_shapes = deepcopy(model.output_shapes)

    return OnnxNet(start_nodes, final_nodes, nodes, output_dict, node_prevs, node_nexts, input_shapes, output_shapes)
end
using Images, ImageIO
using Flux, ONNXNaiveNASflux, NaiveNASflux, NaiveNASlib
using LinearAlgebra
using OpenCV
using CSV
using DataFrames

function square_img(image, bbox, img_size)
    bbox_width = round(Int, bbox["bbox_x2"] - bbox["bbox_x1"])
    bbox_height = round(Int, bbox["bbox_y2"] - bbox["bbox_y1"])
    offset_tl = Int.([bbox["bbox_x1"], bbox["bbox_y1"]] .+ 1)
    h, _ = img_size
    if bbox_width >= bbox_height
        sq_img = zeros(UInt8, bbox_width, bbox_width, 3)
        if bbox_width >= h
            crop_img = image[:, offset_tl[1]: offset_tl[1] + bbox_width - 1, :]
            offset_tl[2] = 1
            bbox_height = h
        elseif offset_tl[2] + bbox_width - 1 >= h
            offset_tl[2] = h + 1 - bbox_width
            bbox_height = bbox_width
            crop_img = image[offset_tl[2] : offset_tl[2] + bbox_width - 1, offset_tl[1] : offset_tl[1] + bbox_width - 1, :]
        else
            crop_img = image[offset_tl[2] : offset_tl[2] + bbox_width - 1, offset_tl[1]+1 : offset_tl[1] + bbox_width - 1, :]
            bbox_height = bbox_width
        end
        sq_img[1:bbox_height, 1:bbox_width, :] = crop_img
    else
        sq_img = zeros(UInt8, bbox_height, bbox_height, 3)
    end
    return sq_img, offset_tl
end

function preprocess_image(img_path, csv_path)
    image = OpenCV.imread(img_path)
    _, w, h = size(image)  # (3, 1920, 1200)
    # CWH, BGR channel for Julia OpenCV, we desire WHCN for Flux model
    image = OpenCV.cvtColor(image, OpenCV.COLOR_BGR2GRAY)
    image = OpenCV.cvtColor(image, OpenCV.COLOR_GRAY2RGB)
    image = permutedims(image, (3, 2, 1))
    
    truth_df = DataFrame(CSV.File(csv_path))
    bbox = truth_df[!, [:bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2]]
    bbox = bbox[parse(Int, split(img_path[1:end-4], '_')[end]) + 1, :]
    sq_image, offset_tl = square_img(image, bbox, (h, w))
    input_size = 256
    sq_image = permutedims(sq_image, (3, 2, 1))  # no need to permute multiple times, will remove it later
    resize_img = OpenCV.resize(sq_image, OpenCV.Size{Int32}(input_size, input_size), interpolation=OpenCV.INTER_AREA)
    resize_img = resize_img / 255.0
    resize_img = permutedims(resize_img, (3, 2, 1))
    return resize_img
end

function get_parallel_chains(comp_vertices, index_more_than_one_outputs)
    function get_chain(vertex)
        m = Any[]
        curr_vertex = vertex
        while length(inputs(curr_vertex)) == 1
            # println("curr vertex ", name(curr_vertex))
            push!(m, layer(curr_vertex))
            curr_vertex = outputs(curr_vertex)[1]
        end
        return Chain(m...), curr_vertex
    end
    outs = outputs(comp_vertices[index_more_than_one_outputs])
    @assert length(outs) == 2
    chain1, vertex_more_than_one_inputs = get_chain(outs[1])
    chain2, _ = get_chain(outs[2])
    @assert occursin("Add", name(vertex_more_than_one_inputs))
    inner_iter = findfirst(v -> name(v) == name(vertex_more_than_one_inputs), comp_vertices)
    if length(chain1) == 0
        return SkipConnection(chain2, (+)), inner_iter
    elseif length(chain2) == 0
        return SkipConnection(chain1, (+)), inner_iter
    else
        return Parallel(+; α = chain1, β = chain2), inner_iter
    end
end

function build_flux_model(onnx_model_path)
    comp_graph = ONNXNaiveNASflux.load(onnx_model_path)

    # find mean value
    model_vec = Any[]
    # sub_vertices = findvertices("/Sub", comp_graph)
    # if !isempty(sub_vertices)
    #     img_mean = inputs(sub_vertices[1])[2]()
    #     println(img_mean)
    #     # println(inputs(vertices(comp_graph)[5])[1]())

    #     push!(model_vec, x -> x .- img_mean)
    # end
    img_mean = reshape([0.48500, 0.45600, 0.40600], (1, 1, 3))
    push!(model_vec, x -> x .- img_mean)

    img_variance = reshape([0.2990, 0.22400, 0.22500], (1, 1, 3))
    push!(model_vec, x -> x ./ img_variance)

    inner_iter = 0
    for (index, vertex) in enumerate(vertices(comp_graph))
        if index < 5 || index <= inner_iter
            continue
        end 
        # println(index, "   ",layer(vertex))
        push!(model_vec, layer(vertex))
        if length(outputs(vertex)) > 1
            # println(name(vertex))
            parallel_chain, inner_iter = get_parallel_chains(vertices(comp_graph), index)
            push!(model_vec, parallel_chain)
        end
    end
    model = Chain(model_vec...)
    Flux.testmode!(model)
    return (model)
end
  
# +++++++++++++++++++ build flux model +++++++++++++++++++
onnx_model_path = "./resnet_model.onnx"
model = build_flux_model(onnx_model_path)
println.(model)
# # +++++++++++++++++++ preprocess image +++++++++++++++++++
img_path = "./test_imgs/AircraftInspection_00000008.png"
csv_path = "./test_imgs/SynthPlane_08.csv"
img = preprocess_image(img_path, csv_path)
input = reshape(img, (size(img)...,1))   # WHCN for Julia, NCHW for Python
input = permutedims(input, (2, 1, 3, 4))
output = model(Float32.(input))
# println(output[1:5,1:5,1,1])

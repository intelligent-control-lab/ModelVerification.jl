using Revise
using ModelVerification
using LazySets
using Plots; default(size = (400,300), legend = :none)
using Flux
using Test
using MLDatasets: CIFAR10, MNIST
using MLUtils: splitobs, DataLoader
import Random
using JLD2
using Profile
using CSV
using ONNX
using Accessors
using BlockDiagonals
using Images, ImageIO
using ONNXNaiveNASflux, NaiveNASflux, .NaiveNASlib
using LinearAlgebra
using OpenCV
using DataFrames
using IJulia
IJulia.installkernel("Julia nodeps", "--depwarn=no")

function build_flux_model(onnx_model_path)
    comp_graph = ONNXNaiveNASflux.load(onnx_model_path)
    model_vec = Any[]
    inner_iter = 0
    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        if length(string(NaiveNASflux.name(vertex))) >= 4 && string(NaiveNASflux.name(vertex))[1:4] == "data"
            continue
        end 
        push!(model_vec, NaiveNASflux.layer(vertex))
        if length(NaiveNASflux.outputs(vertex)) > 1
            parallel_chain, inner_iter = get_parallel_chains(ONNXNaiveNASflux.vertices(comp_graph), index)
            push!(model_vec, parallel_chain)
        end
    end
    model = Chain(model_vec...)
    #Flux.testmode!(model)
    return (model)
end


function square_img(image, bbox, img_size)
    # append blank to make the image a square image
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
    # permute channels, resize, and rescale the image.
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

function build_boeing_flux_model(onnx_model_path)
    # Specifically crafted method for loading Boeing's model
    # This function will append an average pooling layer and flatten layer in the end
    # to convert the problem to a classification verification problem.
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
    comp_graph = ONNXNaiveNASflux.load(onnx_model_path)
    model_vec = Any[]
    inner_iter = 0
    for (index, vertex) in enumerate(ONNXNaiveNASflux.vertices(comp_graph))
        if index < 5 || index <= inner_iter
            continue
        end 
        if string(layer(vertex)) == "#213"
            push!(model_vec, NNlib.relu)
        else
            push!(model_vec, layer(vertex))
        end
        if length(outputs(vertex)) > 1
            # println("name: ", name(vertex))
            parallel_chain, inner_iter = get_parallel_chains(ONNXNaiveNASflux.vertices(comp_graph), index)
            push!(model_vec, parallel_chain)
        end
    end
    # model_vec = model_vec[1:end-1]
    push!(model_vec, Flux.MeanPool((32,32)))
    push!(model_vec, Flux.flatten)
    model = Chain(model_vec...)
    Flux.testmode!(model)
    return (model)
end

function load_image(img_path, csv_path)
    img = preprocess_image(img_path, csv_path)
    input = reshape(img, (size(img)...,1))   # WHCN for Julia, NCHW for Python
    input = permutedims(input, (2, 1, 3, 4))
    img_mean = reshape([0.48500, 0.45600, 0.40600], (1, 1, 3, 1))
    img_variance = reshape([0.2990, 0.22400, 0.22500], (1, 1, 3, 1))
    input = (input .- img_mean) ./ img_variance
    # output = model(Float32.(input));
    return input
end


# create key point error spec
function classification_matrix(n, target)
    A = Matrix{Float64}(I, n, n)
    A[:, target] .= -1
    A = [A[1:target-1, :]; A[target+1:end, :]]
    return A
end
function keypoint_error_spec(y, keypoint_loc, sep)
    mtxs = vec([classification_matrix(sep, t) for t in keypoint_loc])
    A = BlockDiagonal(mtxs)
    return HPolyhedron(A, zeros(size(A,1)))
end
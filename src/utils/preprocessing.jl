
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem)
    return problem
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageStar, problem::Problem)
    return Problem(problem.onnx_model_path, problem.Flux_model, problem.input_shape, init_bound(prop_method, problem.input), problem.output)
end

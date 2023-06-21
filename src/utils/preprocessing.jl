
function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem)
    return problem
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ImageStar, problem::Problem)
    return Problem(problem.model, init_bound(prop_method, problem.input), problem.output)
end
struct ExactReach <: SequentialForwardProp end

struct ExactReachBound <: Bound
    polys::AbstractArray{LazySet}
end

function center(bound::ExactReachBound)
    return sample(bound.polys[1])
end

function prepare_problem(search_method::SearchMethod, split_method::SplitMethod, prop_method::ExactReach, problem::Problem)
    model_info = onnx_parse(problem.onnx_model_path)
    return model_info, Problem(problem.onnx_model_path, problem.Flux_model, init_bound(prop_method, problem.input), problem.output)
end

function init_bound(prop_method::ExactReach, bound::LazySet)
    return ExactReachBound([bound])
end

function check_inclusion(prop_method::ExactReach, model, input::ExactReachBound, reach::ExactReachBound, output::LazySet)
    for bound in reach.polys
        ⊆(bound, output) && continue
        return BasicResult(:violated)
    end
    return BasicResult(:holds)
end

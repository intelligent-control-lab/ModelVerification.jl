
"""
    search_adv_input_bound(search_method::SearchMethod, 
                           split_method::SplitMethod, 
                           prop_method::PropMethod, 
                           problem::Problem;
                           eps = 1e-3)

This function is used to search the maximal input bound that can pass the 
verification (get `:holds`) with the given setting. The search is done by 
binary search on the input bound that is scaled by the given ratio. This 
function is called in `verify` function when `search_adv_bound` is set to 
`true` and the initial verification result is `:unknown` or `:violated`. 

## Arguments
- `search_method` (`SearchMethod`): The search method, such as `BFS`, used to 
    search through the branches. 
- `split_method` (`SplitMethod`): The split method, such as `Bisect`, used to 
    split the branches.
- `prop_method` (`PropMethod`): The propagation method, such as `Ai2`, used to 
    propagate the constraints.
- `problem` (`Problem`): The problem to be verified - consists of a network, 
    input set, and output set.
- `eps` (`Real`): The precision of the binary search. Defaults to 1e-3. 

## Returns
- The maximal input bound that can pass the verification (get `:holds`) with 
    the given setting.
"""
function search_adv_input_bound(search_method::SearchMethod, split_method::SplitMethod, prop_method::PropMethod, problem::Problem;
                        eps = 1e-3)
    l = 0
    r = 1
    while (r-l) > eps
        m = (l+r) / 2
        new_input = scale_set(problem.input, m)
        new_problem = Problem(problem.onnx_model_path, problem.Flux_model, new_input, problem.output)
        res = verify(search_method, split_method, prop_method, new_problem)
        if res.status == :holds
            l = m
            # println("verified ratio: ",m)
        else
            # println("falsified ratio: ",m)
            r = m
        end
    end
    return l, scale_set(problem.input, l)
end

"""
    scale_set(set::Hyperrectangle, ratio)

Scale the hyperrectangle set by the given ratio. The center of the set is 
not changed, but the radius is scaled by the ratio.

## Arguments
- `set` (`Hyperrectangle`): The set to be scaled.
- `ratio` (`Real`): The ratio to scale the set by.

## Returns
- The scaled `Hyperrectangle` set.
"""
function scale_set(set::Hyperrectangle, ratio)
    return Hyperrectangle(center(set), radius_hyperrectangle(set) * ratio)
end

"""
    scale_set(set::ImageConvexHull, ratio)

Scale the image convex hull set by the given ratio. The first image is not 
changed, but the rest of the images are scaled by the ratio. The first image 
is not changed because it acts as the "center" of the set. 

## Arguments
- `set` (`ImageConvexHull`): The set to be scaled.
- `ratio` (`Real`): The ratio to scale the set by.

## Returns
- The scaled `ImageConvexHull` set.
"""
function scale_set(set::ImageConvexHull, ratio)
    new_set = ImageConvexHull(copy(set.imgs))
    new_set.imgs[2:end] = [set.imgs[1] + (img - set.imgs[1]) * ratio for img in set.imgs[2:end]]
    return new_set
end


abstract type ForwardProp <: PropMethod end
abstract type BackwardProp <: PropMethod end
abstract type AdversarialAttack <: PropMethod end

struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope, Star}} <: ForwardProp end
Ai2() = Ai2{Zonotope}()
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Box = Ai2{Hyperrectangle}  

struct StarSet <: ForwardProp
    pre_bound_method::Union{ForwardProp, Nothing}
end
StarSet() = StarSet(nothing)

struct Crown <: ForwardProp 
    bound_lower::Bool
    bound_upper::Bool
end

struct AlphaCrown <: BackwardProp 
    bound_lower::Bool
    bound_upper::Bool
end

struct BetaCrown <: BackwardProp 
    bound_lower::Bool
    bound_upper::Bool
end

struct ImageStar{T<:Union{Star, Zonotope}} <: ForwardProp end
ImageStar() = ImageStar{Star}()
const ImageStarZono = ImageStar{Zonotope}

function prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = Dict(node => Dict() for node in model_info.all_nodes)
    for node in model_info.all_nodes
        push!(batch_info[node], "bounded" => true)
    end
    return init_batch_bound(prop_method, batch_input), batch_output, batch_info
end

function prepare_method(prop_method::StarSet, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = Dict(node => Dict() for node in model_info.all_nodes)
    if hasproperty(prop_method, :pre_bound_method) && !isnothing(prop_method.pre_bound_method)
        pre_batch_bound, pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, batch_output, model_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_bound, pre_batch_out_spec, pre_batch_info)
        for node in model_info.all_nodes
            if haskey(pre_batch_info[node], "bound")
                batch_info[node]["pre_bound"] = pre_batch_info[node]["bound"]
            end
        end
    end
    return init_batch_bound(prop_method, batch_input), batch_output, batch_info
end

function prepare_method(prop_method::Crown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    # prop_method.bound_lower = true
    # prop_method.bound_upper = false
    batch_info = Dict(node => Dict() for node in model_info.all_nodes)
    return init_batch_bound(prop_method, batch_input), get_linear_spec(batch_output), batch_info
end

function prepare_method(prop_method::AlphaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = Dict(node => Dict() for node in model_info.all_nodes)
    prop_method.bound_lower = true
    prop_method.bound_upper = false
    for node in model_info.all_nodes
        bound = AlphaCrownBound(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
        push!(batch_info[node], "bound" => bound)
        push!(batch_info[node], "bounded" => true)
    end
    
    #push!(batch_info[node], "lA" => prop_method.bound_lower ? C : nothing)
    #push!(batch_info[node], "uA" => prop_method.bound_upper ? C : nothing)
    lb = ub = 0 #lb, ub => lower bound, upper bound
    C, batch_size, output_dim, output_shape = preprocess(C)#size(C)=(10, 9, 1) 
    #batch_size = 1, output_dim = 9, output_shape = [-1] 

    return init_batch_bound(prop_method, batch_input), get_linear_spec(batch_output), batch_info
end

function prepare_method(prop_method::BetaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = Dict(node => Dict() for node in model_info.all_nodes)
    prop_method.bound_lower = true
    prop_method.bound_upper = false
    for node in model_info.all_nodes
        push!(batch_info[node], "bounded" => true)
    end
    return init_batch_bound(prop_method, batch_input), get_linear_spec(batch_output), batch_info
end

function preprocess(C)
    batch_size = size(C)[end]
    output_dim = size(C)[2]
    output_shape = [-1]
    return batch_size, output_dim, output_shape
end
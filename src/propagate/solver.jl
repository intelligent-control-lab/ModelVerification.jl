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


function init_start_node_bound(prop_method, batch_input, model_info)
    @assert length(model_info.start_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.start_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input)
    return batch_info
end

function prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, model_info, batch_info)
    return batch_output, batch_info
end

function prepare_method(prop_method::StarSet, batch_input::AbstractVector, batch_output::AbstractVector, model_info, batch_info)
    if hasproperty(prop_method, :pre_bound_method) && !isnothing(prop_method.pre_bound_method)
        pre_batch_info = init_start_node_bound(prop_method.pre_bound_method, batch_input, model_info)
        pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, batch_output, model_info, pre_batch_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_out_spec, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            batch_info[node][:pre_bound] = pre_batch_info[prev_node][:bound]
        end
    end
    return batch_output, batch_info
end

function prepare_method(prop_method::Crown, batch_input::AbstractVector, batch_output::AbstractVector, model_info, batch_info)
    # prop_method.bound_lower = true
    # prop_method.bound_upper = false
    batch_info[model_info.start_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input)
    return get_linear_spec(batch_output), batch_info
end

function prepare_method(prop_method::AlphaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info, batch_info)
    for node in model_info.all_nodes
        bound = AlphaCrownBound(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
        push!(batch_info[node], :bound => bound)
    end
    return get_linear_spec(batch_output), batch_info
end

function prepare_method(prop_method::BetaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info, batch_info)
    return get_linear_spec(batch_output), batch_info
end

function preprocess(C)
    batch_size = size(C)[end]
    output_dim = size(C)[2]
    output_shape = [-1]
    return batch_size, output_dim, output_shape
end
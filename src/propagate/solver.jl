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
    pre_bound_method::Union{ForwardProp, Nothing}
    bound_lower::Bool
    bound_upper::Bool
    optimizer
    trian_iteration::Int
end

struct BetaCrown <: BackwardProp 
    bound_lower::Bool
    bound_upper::Bool
end

struct ImageStar{T<:Union{Star, Zonotope}} <: ForwardProp end
ImageStar() = ImageStar{Star}()
const ImageStarZono = ImageStar{Zonotope}


function init_start_node_bound(prop_method::ForwardProp, batch_input, model_info)
    @assert length(model_info.start_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.start_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input)
    return batch_info
end


function init_start_node_bound(prop_method::AlphaCrown, batch_input, model_info)
    @assert length(model_info.start_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
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
    batch_info[model_info.start_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input)
    return get_linear_spec(batch_output), batch_info
end

function prepare_method(prop_method::AlphaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info, batch_info)
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
    
    #initialize alpha 
    for node in model_info.activation_nodes
        init_alpha(model_info.node_layer[node], node, batch_info)
    end

    for node in model_info.all_nodes
        batch_info[node][:beta] = 1
        batch_info[node][:weight_ptb] = false
        batch_info[node][:bias_ptb] = false
    end
    
    batch_info[:Alpha_Lower_Layer_node] = []#store the order of the node which has AlphaLayer
    batch_info[:batch_size] = length(batch_input)
    linear_spec = get_linear_spec(batch_output)
    batch_info[:spec_number] = size(linear_spec.A)[end]
    init_A_bias(prop_method, batch_input, batch_info)
    batch_info[model_info.final_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input)
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

function init_A_bias(prop_method::AlphaCrown, batch_input, batch_info)
    # batch_input : list of Hyperrectangle
    batch_size = length(batch_input)
    batch_info[:batch_size] = batch_size
    n = dim(batch_input[1])
    I = Matrix{Float64}(LinearAlgebra.I(n))
    Z = zeros(n)
    A = repeat(I, outer=(1, 1, batch_size))
    b = repeat(Z, outer=(1, 1, batch_size))
    batch_info[:init_lower_A_bias] = [A, b]
    batch_info[:init_upper_A_bias] = [A, b]
end
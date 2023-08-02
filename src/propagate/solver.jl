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

mutable struct AlphaCrown <: BackwardProp 
    pre_bound_method::Union{ForwardProp, Nothing}
    bound_lower::Bool
    bound_upper::Bool
    optimizer
    train_iteration::Int
end

mutable struct BetaCrown <: BackwardProp 
    bound_lower::Bool
    bound_upper::Bool
end

struct ImageStar <: ForwardProp 
    pre_bound_method::Union{ForwardProp, Nothing}
end
ImageStar() = ImageStar(nothing)

struct ImageZono <: ForwardProp end

function init_propagation(prop_method::ForwardProp, batch_input, batch_output, model_info)
    @assert length(model_info.start_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.start_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input, batch_output)
    return batch_info
end

function init_propagation(prop_method::BackwardProp, batch_input, batch_output, model_info)
    @assert length(model_info.final_nodes) == 1
    batch_info = Dict{Any, Any}(node => Dict() for node in model_info.all_nodes)
    batch_info[model_info.final_nodes[1]][:bound] = init_batch_bound(prop_method, batch_input, batch_output)
    return batch_info
end

function prepare_method(prop_method::PropMethod, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, batch_output, model_info)
    return batch_output, batch_info
end

function prepare_method(prop_method::Union{StarSet, ImageStar}, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, batch_output, model_info)
    if hasproperty(prop_method, :pre_bound_method) && !isnothing(prop_method.pre_bound_method)
        batch_input = init_batch_bound(prop_method.pre_bound_method, batch_input, batch_output)
        pre_batch_info = init_propagation(prop_method.pre_bound_method, batch_input, batch_output, model_info)
        pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, batch_output, model_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
        for node in model_info.activation_nodes
            @assert length(model_info.node_prevs[node]) == 1
            prev_node = model_info.node_prevs[node][1]
            batch_info[node][:pre_bound] = pre_batch_info[prev_node][:bound]
        end
    end
    return batch_output, batch_info
end

function prepare_method(prop_method::Crown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    batch_info = init_propagation(prop_method, batch_input, batch_output, model_info)
    return get_linear_spec(batch_output), batch_info
end

function prepare_method(prop_method::AlphaCrown, batch_input::AbstractVector, batch_output::AbstractVector, model_info)
    
    out_specs = get_linear_spec(batch_output)
    prop_method.bound_lower = out_specs.is_complement ? true : false
    prop_method.bound_upper = out_specs.is_complement ? false : true
    batch_info = init_propagation(prop_method, batch_input, out_specs, model_info)
    
    batch_info[:spec_A_b] = [out_specs.A, .-out_specs.b] # spec_A x < spec_b  ->  A x + b < 0, need negation
    batch_info[:init_upper_A_b] = [out_specs.A, .-out_specs.b]

    # After conversion, we only need to decide if lower bound of spec_A y-spec_b > 0 or if upper bound of spec_A y - spec_b < 0
    # The new out is spec_A*y-b, whose dimension is spec_dim x batch_size.
    # Therefore, we set new_spec_A: 1(new_spec_dim) x original_spec_dim x batch_size, new_spec_b: 1(new_spec_dim) x batch_size,
    # spec_dim, out_dim, batch_size = size(out_specs.A)
    # out_specs = LinearSpec(ones((1, spec_dim, batch_size)), zeros(1, batch_size), out_specs.is_complement)

    if hasproperty(prop_method, :pre_bound_method) && !isnothing(prop_method.pre_bound_method)
        pre_batch_out_spec, pre_batch_info = prepare_method(prop_method.pre_bound_method, batch_input, batch_output, model_info)
        pre_batch_bound, pre_batch_info = propagate(prop_method.pre_bound_method, model_info, pre_batch_info)
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
    # init_A_b(prop_method, batch_input, batch_info)

    return out_specs, batch_info
end

function preprocess(C)
    batch_size = size(C)[end]
    output_dim = size(C)[2]
    output_shape = [-1]
    return batch_size, output_dim, output_shape
end

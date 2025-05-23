"""
    read_nnet(fname::String; last_layer_activation = Id())

Read in neural net from a `.nnet` file and return Network struct.
The `.nnet` format is borrowed from [NNet](https://github.com/sisl/NNet).
The format assumes all hidden layers have ReLU activation.
Keyword argument `last_layer_activation` sets the activation of the last
layer, and defaults to `Id()`, (i.e. a linear output layer).

## Arguments
- `fname` (`String`): String path to the `.nnet` file.
- `last_layer_activation`: Keyword argument that sets the activation of the last 
    layer which defaults to `Id()`.

## Returns
- A vector of layers saved as `Network`.
"""
function read_nnet(fname::String; last_layer_activation = Id())
    f = open(fname)
    line = readline(f)
    while occursin("//", line) #skip comments
        line = readline(f)
    end
    # number of layers
    nlayers = parse(Int64, split(line, ",")[1])
    # read in layer sizes
    layer_sizes = parse.(Int64, split(readline(f), ",")[1:nlayers+1])
    # read past additonal information
    for i in 1:5
        line = readline(f)
    end
    # i=1 corresponds to the input dimension, so it's ignored
    layers = Layer[read_layer(dim, f) for dim in layer_sizes[2:end-1]]
    push!(layers, read_layer(last(layer_sizes), f, last_layer_activation))

    return Network(layers)
end

"""
    read_layer(output_dim::Int64, f::IOStream, act = ReLU())

Read in layer from `.nnet` file and return a `Layer` containing its 
weights & biases. Optional argument `act` sets the activation function for the 
layer.

## Arguments
- `output_dim` (Int64): Output dimension of the layer.
- `f` (`IOStream`): IO stream of the `.nnet` file.
- `act`: Optional argument specifying the activation function of the layer. 
    Defaults to `ReLU()`.

## Returns
- `Layer` containing the weights and bias values (and the activation function 
    of the layer).
"""
function read_layer(output_dim::Int64, f::IOStream, act = ReLU())

    rowparse(splitrow) = parse.(FloatType[], splitrow[findall(!isempty, splitrow)])
    # first read in weights
    W_str_vec = [rowparse(split(readline(f), ",")) for i in 1:output_dim]
    weights = vcat(W_str_vec'...)
    # now read in bias
    bias_string = [split(readline(f), ",")[1] for j in 1:output_dim]
    bias = rowparse(bias_string)
    # activation function is set to ReLU as default
    return Layer(weights, bias, act)
end

"""
    to_comment(txt)

Prepend `//` to each line of a string.
"""
to_comment(txt) = "//"*replace(txt, "\n"=>"\n//")

"""
    print_layer(file::IOStream, layer)

Print to `file` an object implementing `weights(layer)` and `bias(layer)`.

## Arguments
- `file` (`IOStream`): IO stream of the target `.nnet` file.
- `layer`: Layer to be transcribed to `file`.
"""
function print_layer(file::IOStream, layer)
   print_row(W, i) = println(file, join(W[i,:], ", "), ",")
   W = layer.weights
   b = layer.bias
   [print_row(W, row) for row in axes(W, 1)]
   [println(file, b[row], ",") for row in axes(W, 1)]
end

"""
    print_header(file::IOStream, network; header_text="")

The NNet format has a particular header containing information about the 
network size and training data. `print_header` does not take training-related 
information into account (subject to change).

## Arguments
- `file` (`IOStream`): IO stream of the target `.nnet` file.
- `network`: Network to be transcribed to `file`.
- `header_text`: Optional header text that comes before the network information. 
    Defaults to an empty string.
"""
function print_header(file::IOStream, network; header_text="")
   println(file, to_comment(header_text))
   layer_sizes = [size(layer.weights, 1) for layer in network.layers] # doesn't include the input layer
   pushfirst!(layer_sizes, size(network.layers[1].weights, 2)) # add the input layer

   # num layers, num inputs, num outputs, max layer size
   num_layers = length(network.layers)
   num_inputs = layer_sizes[1]
   num_outputs = layer_sizes[end]
   max_layer = maximum(layer_sizes[1:end-1]) # chop off the output layer for the maximum,
   println(file, join([num_layers, num_inputs, num_outputs, max_layer], ", "), ",")
   #layer sizes input, ..., output
   println(file, join(layer_sizes, ", "), ",")
   # empty
   println(file, "This line extraneous")
   # minimum vals of inputs
   println(file, string(join(fill(-floatmax(Float16), num_inputs), ","), ","))
   # maximum vals of inputs
   println(file, string(join(fill(floatmax(Float16), num_inputs), ","), ","))
   # mean vals of inputs + 1 for output
   println(file, string(join(fill(0.0, num_inputs+1), ","), ","))
   # range vals of inputs + 1 for output
   println(file, string(join(fill(1.0, num_inputs+1), ","), ","))
   return nothing
end

"""
    write_nnet(filename, network; header_text)

Write `network` to `filename.nnet`.
Note: Does not perform safety checks on inputs, so use with caution.

Based on python code at https://github.com/sisl/NNet/blob/master/utils/writeNNet.py
and follows .nnet format given here: https://github.com/sisl/NNet.

## Arguments
- `outfile`: String name of the `.nnet` file.
- `network`: Network to be transcribed to `outfile.nnet`.
- `header_text`: Optional header text that comes before the network information.
"""
function write_nnet(outfile, network; header_text="Default header text.\nShould replace with the real deal.")
    name, ext = splitext(outfile)
    outfile = name*".nnet"
    open(outfile, "w") do f
        print_header(f, network, header_text=header_text)
        for layer in network.layers
            print_layer(f, layer)
        end
    end
    nothing
end
"""
    compute_output(nnet::Network, input)

Propagates a given vector through a `Network` and computes the output.

## Arguments
- `nnet` (`Network`): Network to be propagated.
- `input`: Vector to be propagated through `nnet`.

## Returns
- `Vector` of the output of `nnet` given `input`.
"""
function compute_output(nnet::Network, input)
    curr_value = input
    for layer in nnet.layers # layers does not include input layer (which has no weights/biases)
        curr_value = layer.activation(affine_map(layer, curr_value))
    end
    return curr_value # would another name be better?
end

"""
    get_activation(L::Layer{ReLU}, x::Vector)
    
Finds the activation pattern of a vector `x` subject to the activation function 
given by the layer `L`. Returns a Vector{Bool} where `true` denotes the node is 
"active". In the sense of ReLU, this would be `x[i] >= 0`.

## Arguments
- `L` (`Layer{ReLU}`): Layer with ReLU activation function.
- `x` (`Vector`): Vector to be propagated through the ReLU activation function.

## Returns
- `Vector{Bool}` where `true` denotes the node is element-wise "active".
"""
get_activation(L::Layer{ReLU}, x::Vector) = x .>= 0.0

"""
    get_activation(L::Layer{Id}, args...)
    
Finds the activation pattern of a vector `x` subject to the activation function 
given by the layer `L`. Returns a Vector{Bool} where `true` denotes the node is 
"active". In the sense of Identity, this would be a vector of `true`'s for all 
nodes in the layer `L`.

## Arguments
- `L` (`Layer{Id}`): Layer with Identity activation function.
- `x` (`Vector`): Vector to be propagated through the Identity activation 
    function.
    
## Returns
- `Vector{Bool}` where `true` denotes the node is element-wise "active".
"""
get_activation(L::Layer{Id}, args...) = trues(n_nodes(L))

"""
    get_activation(nnet::Network, x::Vector{FloatType[]})

Given a network, find the activation pattern of all neurons at a given point x.
Returns Vector{Vector{Bool}}. Each Vector{Bool} refers to the activation pattern 
of a particular layer.

## Arguments
- `nnet` (`Network`): Network to be propagated.
- `x` (`Vector{FloatType[]}`): Vector to be propagated through `nnet`.

## Returns
- `Vector{Vector{Bool}}` where each Vector{Bool} refers to the activation 
    pattern of a particular layer.
"""
function get_activation(nnet::Network, x::Vector{FloatType[]})
    act_pattern = Vector{Vector{Bool}}(undef, length(nnet.layers))
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = affine_map(layer, curr_value)
        act_pattern[i] = get_activation(layer, curr_value)
        curr_value = layer.activation(curr_value)
    end
    return act_pattern
end

"""
    get_activation(nnet::Network, input::Hyperrectangle)

Given a network, find the activation pattern of all neurons for a given input 
set. Assume ReLU activation function for all layers. This function first 
computes the node-wise bounds of the input set, and then computes the 
activation pattern using `get_activation(nnet, bounds)`.

## Arguments
- `nnet` (`Network`): Network to be propagated, with the activation function 
    assumed to be ReLU for all layers.
- `input` (`Hyperrectangle`): Input set to be propagated through `nnet`, 
        represented as a `Hyperrectangle`.

## Returns
- `Vector{Vector{Bool}}` where each Vector{Bool} refers to the activation 
    pattern of a particular layer. Each element in each `Vector{Bool}` specifies 
    the activation pattern of a particular neuron. `1` means activated, `0` 
    means undetermined, and `-1` means not activated.
"""
function get_activation(nnet::Network, input::Hyperrectangle)
    bounds = get_bounds(nnet, input, before_act = true)
    return get_activation(nnet, bounds)
end

"""
    get_activation(nnet::Network, bounds::Vector{Hyperrectangle})

Given a network, find the activation pattern of all neurons given the node-wise 
bounds. Assume ReLU activation function for all layers. Assume pre-activation 
bounds where the bounds on the input are given by the first hyperrectangle, the 
first hidden layer by the second hyperrectangle, and so on.

## Arguments
- `nnet` (`Network`): Network to be propagated, with the activation function 
    assumed to be ReLU for all layers.
- `bounds` (`Vector{Hyperrectangle}`): Vector of node-wise bounds, where the 
    bounds on the input are given by the first hyperrectangle, the first hidden 
    layer by the second hyperrectangle, and so on.

## Returns
- `Vector{Vector{Bool}}` where each Vector{Bool} refers to the activation 
    pattern of a particular layer. Each element in each `Vector{Bool}` specifies 
    the activation pattern of a particular neuron. `1` means activated, `0` 
    means undetermined, and `-1` means not activated.
"""
function get_activation(nnet::Network, bounds::Vector{Hyperrectangle})
    act_pattern = Vector{Vector{Int}}(undef, length(nnet.layers))
    for (i, layer) in enumerate(nnet.layers)
        act_pattern[i] = get_activation(layer, bounds[i+1])
    end
    return act_pattern
end

"""
    get_activation(L::Layer{ReLU}, bounds::Hyperrectangle)

Given a layer, find the activation pattern of all neurons in the layer given the 
node-wise bounds. Assume ReLU activation function for the given layer. Assume 
bounds is the pre-activation bounds for each ReLU in the layer.

## Arguments
- `L` (`Layer{ReLU}`): Layer to be propagated, with the activation function 
    assumed to be ReLU.
- `bounds` (`Hyperrectangle`): Node-wise pre-activation bounds for the layer.

## Returns
- `Vector{Bool}` where each element refers to the activation pattern of a 
    particular neuron. `1` means activated, `0` means undetermined, and `-1` 
    means not activated.
"""
function get_activation(L::Layer{ReLU}, bounds::Hyperrectangle)
    lower = low(bounds)
    upper = high(bounds)
    act_pattern = zeros(n_nodes(L))
    for j in 1:n_nodes(L) # For evey node
        if lower[j] > 0.0
            act_pattern[j] = 1
        elseif upper[j] < 0.0
            act_pattern[j] = -1
        end
    end
    return act_pattern
end

"""
    get_gradient(nnet::Network, x::Vector)

Given a network, find the gradient for the input x. The function propagates 
through the layers of the network, and computes the gradient at each layer. 
The gradient at each layer is computed by multiplying the gradient at the 
previous layer with the gradient of the current layer.

## Arguments
- `nnet` (`Network`): Network to be propagated.
- `x` (`Vector`): Vector to be propagated through `nnet`.

## Returns
- `Matrix` of the gradient for the input `x` after propagating through `nnet`.
"""
function get_gradient(nnet::Network, x::Vector)
    z = x
    gradient = Matrix(1.0I, length(x), length(x))
    for (i, layer) in enumerate(nnet.layers)
        z_hat = affine_map(layer, z)
        σ_gradient = act_gradient(layer.activation, z_hat)
        gradient = Diagonal(σ_gradient) * layer.weights * gradient
        z = layer.activation(z_hat)
    end
    return gradient
end

"""
    act_gradient(act::ReLU, z_hat::Vector)

Compute the gradient of an ReLU activation function at point z_hat. For each 
element in `z_hat`, if `z_hat[i] > 0`, then `act_gradient[i] = 1`, else 
`act_gradient[i] = 0`.

## Arguments
- `act` (`ReLU`): ReLU activation function.
- `z_hat` (`Vector`): Vector to be propagated through `act`.

## Returns
- `Vector` of the gradient of `act` at `z_hat`. Each element in the vector 
    corresponds to the gradient of a particular neuron.
"""
act_gradient(act::ReLU, z_hat::Vector) = z_hat .>= 0.0

"""
    act_gradient(act::Id, z_hat::Vector)

Compute the gradient of an Identity activation function at point z_hat. For 
each element in `z_hat`, `act_gradient[i] = 1`. 

## Arguments
- `act` (`Id`): Identity activation function.
- `z_hat` (`Vector`): Vector to be propagated through `act`.

## Returns
- `Vector` of the gradient of `act` at `z_hat`. Each element in the vector 
    corresponds to the gradient of a particular neuron. 
"""
act_gradient(act::Id, z_hat::Vector) = trues(length(z_hat))

"""
    relaxed_relu_gradient(l::Real, u::Real)

Return the relaxed slope of a ReLU activation based on its lower and upper 
bounds. The relaxed ReLU function allows for a smooth approximation of the 
gradient of the ReLU function. The relaxed ReLU function is defined as follows: 
- `f'(x) = 0` if `upper-bound < 0`, 
- `f'(x) = 1` if `lower-bound > 0`, 
- and `f'(x) = x/(u-l)` if `lower-bound < x < upper-bound` which is the slope of 
    the line connecting the points `(l, ReLU(l))` and `(u, ReLU(u))`.

This provides a differentiable approximation of the ReLU function within the 
interval [l, u].

## Arguments
- `l` (`Real`): Lower-bound of the input to the ReLU activation function.
- `u` (`Real`): Upper-bound of the input to the ReLU activation function.

## Returns
- `0.0` if `u <= 0.0`, 
- `1.0` if `l >= 0.0`,
- `u/(u-l)` otherwise.
"""
function relaxed_relu_gradient(l::Real, u::Real)
    u <= 0.0 && return 0.0
    l >= 0.0 && return 1.0
    return u / (u - l)
end

"""
    act_gradient_bounds(nnet::Network, input::AbstractPolytope)

Compute the bit-wise bounds on the gradient post activation operation given an 
input  set. Currently only support ReLU activation function. It first calculates 
the bounds of the input for each layer using `get_bounds` function (this 
function propagates through each layer and computes the bounds of each layer). 
Then, it computes the bit-wise lower and upper gradient for each layer using 
`act_gradient`. The final output is a tuple of two vectors, where each vector is 
a vector of bit-vectors. Each element in the vector corresponds to the gradient 
of a particular neuron which can be either `0` (not activated) or 
`1` (activated).

## Arguments
- `nnet` (`Network`): Network to be propagated.
- `input` (`AbstractPolytope`): Input set to be propagated through `nnet`.

## Returns
- `LΛ, UΛ::NTuple{2, Vector{BitVector}}`: lower and upper bit-wsie bounds on the 
    activation gradients for each layer.
"""
function act_gradient_bounds(nnet::Network, input::AbstractPolytope)
    # get the pre-activation bounds, and get rid of the input set
    bounds = get_bounds(nnet, input, before_act=true)
    popfirst!(bounds)

    LΛ = Vector{BitVector}(undef, 0)
    UΛ = Vector{BitVector}(undef, 0)
    for (i, layer) in enumerate(nnet.layers)
        l = act_gradient(layer.activation, low(bounds[i]))
        u = act_gradient(layer.activation, high(bounds[i]))
        push!(LΛ, l)
        push!(UΛ, u)
    end
    return (LΛ, UΛ)
end

"""
    get_gradient_bounds(nnet::Network, input::AbstractPolytope)

Get lower and upper bounds on network gradient for given gradient bounds on 
activations, or given an input set.
It first calculates the bit-wise lower and upper gradient bounds for each layer 
using  `act_gradient_bounds`. Then, it computes the gradient bounds of the 
entire network for the weights using `get_gradient_bounds`.

## Arguments
- `nnet` (`Network`): Network to be propagated.
- `input` (`AbstractPolytope`): Input set to be propagated through `nnet`.

## Returns
- `(LG, UG)`: `NTuple{2, Matrix{FloatType[]}` of the lower and upper bounds of the 
    entire network.
"""
function get_gradient_bounds(nnet::Network, input::AbstractPolytope)
    LΛ, UΛ = act_gradient_bounds(nnet, input)
    return get_gradient_bounds(nnet, LΛ, UΛ)
end

"""
    get_gradient_bounds(nnet::Network, LΛ::Vector{AbstractVector}, 
                        UΛ::Vector{AbstractVector})

Given bit-wise lower and upper gradient bounds for each layer (`LΛ` and `UΛ`), 
compute the lower and upper bounds of the entire network for the weights of the 
layers. 

## Arguments
- `nnet` (`Network`): Network to be propagated.
- `LΛ` (`Vector{AbstractVector}`): Vector of bit-wise lower gradient bounds for 
    each layer. 
- `UΛ` (`Vector{AbstractVector}`): Vector of bit-wise upper gradient bounds for
    each layer.    

## Returns
- `(LG, UG)`: `NTuple{2, Matrix{FloatType[]}` of the lower and upper bounds of the 
    entire network.
"""
function get_gradient_bounds(nnet::Network, LΛ::Vector{<:AbstractVector}, UΛ::Vector{<:AbstractVector})
    n_input = size(nnet.layers[1].weights, 2)
    LG = Matrix(1.0I, n_input, n_input)
    UG = Matrix(1.0I, n_input, n_input)
    for (i, layer) in enumerate(nnet.layers)
        LG_hat, UG_hat = interval_map(layer.weights, LG, UG)
        LG = Diagonal(LΛ[i]) * max.(LG_hat, 0) + Diagonal(UΛ[i]) * min.(LG_hat, 0)
        UG = Diagonal(LΛ[i]) * min.(UG_hat, 0) + Diagonal(UΛ[i]) * max.(UG_hat, 0)
    end
    return (LG, UG)
end

"""
    interval_map(W::Matrix, l::AbstractVecOrMat, u::AbstractVecOrMat)

Simple linear mapping of the weights on intervals.
`L, U := ([W]₊*l + [W]₋*u), ([W]₊*u + [W]₋*l)`

## Arguments
- `W` (`AbstractMatrix`): Matrix of weights.
- `l` (`AbstractVecOrMat`): Vector or matrix of lower bounds.
- `u` (`AbstractVecOrMat`): Vector or matrix of upper bounds.

## Returns
- `(l_new, u_new)`: New bounds after the linear mapping.
"""
function interval_map(W::AbstractMatrix{N}, l::AbstractVecOrMat, u::AbstractVecOrMat) where N
    l_new = max.(W, zero(N)) * l + min.(W, zero(N)) * u
    u_new = max.(W, zero(N)) * u + min.(W, zero(N)) * l
    return (l_new, u_new)
end

"""
    get_bounds(nnet::Network, input; before_act::Bool = false)

Computes node-wise bounds given a input set. The optional last
argument determines whether the bounds are pre- or post-activation.

## Arguments
- `nnet` (`Network`): Network to be propagated.
- `input` (`AbstractPolytope`): Input set to be propagated through `nnet`.
- `before_act`: Optional argument that determines whether the bounds are 
    pre- or post-activation. Defaults to `false`.

## Returns
- `Vector{Hyperrectangle}`: bounds for all nodes. `bounds[1]` is the 
    input set overapproximated with a `Hyperrectangle`.
"""
function get_bounds(nnet::Network, input; before_act::Bool = false) # NOTE there is another function by the same name in convDual. Should reconsider dispatch
    input = overapproximate(input, Hyperrectangle)
    bounds = Vector{Hyperrectangle}(undef, length(nnet.layers) + 1)
    bounds[1] = input
    b = input
    for (i, layer) in enumerate(nnet.layers)
        if before_act
            bounds[i+1] = approximate_affine_map(layer, b)
            b = approximate_act_map(layer, bounds[i+1])
        else
            b = approximate_affine_map(layer, bounds[i])
            bounds[i+1] = approximate_act_map(layer, b)
        end
    end
    return bounds
end

"""
    get_bounds(problem::Problem; kwargs...)

Compute node-wise bounds for a given `Problem` using `get_bounds(nnet, input)`.

## Arguments
- `problem` (`Problem`): Problem to be propagated.
- `kwargs`: Keyword arguments to be passed to `get_bounds(nnet, input)` such as 
    the optional boolean argument `before_act`.

## Returns
- `Vector{Hyperrectangle}`: bounds for all nodes. `bounds[1]` is the input set.
"""
get_bounds(problem::Problem; kwargs...) = get_bounds(problem.network, problem.input; kwargs...)

# """
#     affine_map(layer, x)

# Compute W*x ⊕ b for a vector or LazySet `x`
# """
# affine_map(layer::Layer, x) = layer.weights*x + layer.bias
# function affine_map(layer::Layer, x::LazySet)
#     LazySets.affine_map(layer.weights, x, layer.bias)
# end


"""
   approximate_affine_map(layer, input::Hyperrectangle)

Returns a Hyperrectangle overapproximation of the affine map of the input.

## Arguments
- `layer` (`Layer`): Layer to be propagated for the affine map.
- `input` (`Hyperrectangle`): Input set to be propagated through `layer`.

## Returns
- `Hyperrectangle` overapproximation of the affine map of the input.
"""
function approximate_affine_map(layer::Layer, input::Hyperrectangle)
    c = affine_map(layer, input.center)
    r = abs.(layer.weights) * input.radius
    return Hyperrectangle(c, r)
end

"""
    approximate_act_map(act::ActivationFunction, input::Hyperrectangle)

Returns a Hyperrectangle overapproximation of the activation map of the input.
`act` must be monotonic.

## Arguments
- `act` (`ActivationFunction`): Activation function to be propagated.
- `input` (`Hyperrectangle`): Input set to be propagated through `act`.

## Returns
- `Hyperrectangle` overapproximation of the activation map of the input.
"""
function approximate_act_map(act::ActivationFunction, input::Hyperrectangle)
    β    = act.(input.center)
    βmax = act.(high(input))
    βmin = act.(low(input))
    c    = (βmax + βmin)/2
    r    = (βmax - βmin)/2
    return Hyperrectangle(c, r)
end

"""
    approximate_act_map(layer::Layer, input::Hyperrectangle)

Returns a Hyperrectangle overapproximation of the activation map of the input 
for the given layer. The activation function of the layer must be monotonic. 

## Arguments
- `layer` (`Layer`): Layer to be propagated for the activation map.
- `input` (`Hyperrectangle`): Input set to be propagated through `layer`.

## Returns
- `Hyperrectangle` overapproximation of the activation map of the input.
"""
approximate_act_map(layer::Layer, input::Hyperrectangle) = approximate_act_map(layer.activation, input)

"""
    UnboundedInputError <: Exception

Exception thrown when an input set is unbounded.

## Fields
- `msg` (`String`): Error message.
"""
struct UnboundedInputError <: Exception
    msg::String
end
Base.showerror(io::IO, e::UnboundedInputError) = print(io, msg)

"""
    isbounded(input)

Check if input set is bounded. If the `input` is of type `HPolytope`, then 
`LazySets.isbounded` converts the `HPolytope` to a `HPolyhedron` and checks if 
that is bounded. Otherwise, `LazySets.isbounded` is called directly.

## Arguments
- `input`: Input set to be checked for boundedness.

## Returns
- `true` if `input` is bounded, `false` otherwise.
"""
function isbounded(input)
    if input isa HPolytope
        return LazySets.isbounded(input, false)
    else
        return LazySets.isbounded(input)
    end
end

"""
    is_hypercube(set::Hyperrectangle)

Check if `set` is a is_hypercube. This is done by checking if all the radii of 
the `Hyperrectangle` are equal in all directions.

## Arguments
- `set` (`Hyperrectangle`): Set to be checked for hypercube-ness.

## Returns
- `true` if `set` is a hypercube, `false` otherwise.
"""
is_hypercube(set::Hyperrectangle) = all(iszero.(set.radius .- set.radius[1]))

"""
    is_halfspace_equivalent(set)

Check if `set` is halfspace equivalent. This is done by checking if the number 
of constraints in the `set` is equal to 1.

## Arguments
- `set`: Set to be checked for halfspace equivalence.

## Returns
- `true` if `set` is halfspace equivalent, `false` otherwise.
"""
is_halfspace_equivalent(set) = length(constraints_list(set)) == 1

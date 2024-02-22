"""
    Spec

Abstract super-type for input-output specifications.
"""
abstract type Spec end

"""
    ImageConvexHull <: Spec

Convex hull for images used to specify safety property for images. 
It is the smallest convex polytope that contains all the images given in the `imgs` array.

## Fields
- `imgs` (`AbstractArray`): List of images in `AbstractArray`. Image is 
    represented as a matrix of height x weight x channels.
"""
struct ImageConvexHull <: Spec
    # spec: A x - b <= 0 is the safe set or unsafe set
    imgs::AbstractArray # list of images (h,w,c)
end


function center(bound::ImageConvexHull)
    return sum(bound.imgs) ./ length(bound.imgs)
end

"""
    ReLUConstraints

A mutable structure for storing information related to the constraints of a ReLU 
(Rectified Linear Unit) activation function in a neural network.

## Fields
- `idx_list`: A list of indices. 
- `val_list`: A list of values corresponding to the indices in `idx_list`. 
- `not_splitted_mask`: A mask indicating which elements in `idx_list` and 
    `val_list` have not been split. This is used in the context of a piecewise 
    linear approximation of the ReLU function, where the input space is split 
    into regions where the function is linear.
- `history_split`: A record of the splits that have been performed. 
"""
mutable struct ReLUConstraints
    idx_list
    val_list
    not_splitted_mask
    history_split
end

"""
    ReLUConstrainedDomain <: Spec

A mutable structure for storing specifications related to the ReLU 
(Rectified Linear Unit) activation function in a neural network.

## Fields
- `domain`: A geometric specification representing the domain of the ReLU 
    function.
- `all_relu_cons`: A dictionary of ReLU constraints for each node in the 
    network.
"""
mutable struct ReLUConstrainedDomain <: Spec
    # S_dict : {node => [idx_list, val_list, not_splitted_mask, history_S]}, such that we can do the following when propagate relu
    domain
    all_relu_cons::Dict{String, ReLUConstraints}
end

"""
    ImageLinfBall

A mutable structure for storing information related to the constraints of a 
L-infinity ball for images.

## Fields
- `lb`: Lower bound of the ball.
- `ub`: Upper bound of the ball.
"""
struct ImageLinfBall <: Spec
    lb::AbstractArray # (h,w,c)
    ub::AbstractArray # (h,w,c)
end

"""
    get_image_linf_spec(lb, ub, img_size)

Given a lower bound `lb`, an upper bound `ub`, and the size of the image, 
returns a `ImageZonoBound` structure.

## Arguments
- `lb`: Lower bound of the image.
- `ub`: Upper bound of the image.
- `img_size`: Size of the image.

## Returns
- `ImageZonoBound` structure.
"""
function get_image_linf_spec(lb, ub, img_size)
    h = Hyperrectangle(low = lb, high = ub)
    cen = reshape(center(h), (img_size...,1))
    gen = reshape(genmat(h), (img_size...,length(lb)))
    return ImageZonoBound(cen, gen)
end

"""
    LinearSpec <: Spec

Safety specification defined as the set ``\\{ x: x = A x - b ≤ 0 \\}``.

## Fields
- `A` (`AbstractArray{Float64, 3}`): Normal dierction of size 
    `spec_dim x out_dim x batch_size`.
- `b` (`AbstractArray{Float64, 2}`): Constraints of size 
    `spec_dim x batch_size`.
- `is_complement` (`Bool`): Boolean flag for whether this specification is a 
    complement or not.
"""
struct LinearSpec <: Spec 
    # spec: A x - b <= 0 is the safe set or unsafe set
    A::AbstractArray{Float64, 3} # spec_dim x out_dim x batch_size
    b::AbstractArray{Float64, 2} # spec_dim x batch_size
    is_complement::Bool
end

"""
    InputSpec

Input specification can be of any type supported by `LazySet` or `ImageConvexHull`.
"""
const InputSpec = Union{LazySet, ImageConvexHull}

"""
    OutputSpec

Output specification can be of any type supported by `LazySet` or `LinearSpec`.
"""
const OutputSpec = Union{LazySet, LinearSpec}

"""
    get_size(input_spec::LazySet)

Given a `LazySet`, it determines the size of the set.
"""
get_size(input_spec::LazySet) = size(LazySets.center(input_spec))

"""
    get_size(input_spec::ImageConvexHull)

Given an `ImageConvexHull`, it determines the size of the image.
"""
get_size(input_spec::ImageConvexHull) = size(input_spec.imgs[1])

"""
    get_linear_spec(batch_out_set::AbstractVector)

Retrieves the linear specifications of the batch of output sets and returns
a `LinearSpec` structure. 

## Arguments
- `batch_out_set` (`AbstractVector`): Batch of output sets.

## Returns
- `LinearSpec` of the batch of output sets.
"""
function get_linear_spec(batch_out_set::AbstractVector)
    max_spec_num = maximum([length(constraints_list(o)) for o in batch_out_set])
    out_spec_A = zeros(max_spec_num, dim(batch_out_set[1]), length(batch_out_set)) # spec_dim x out_dim x batch_size
    out_spec_b = zeros(max_spec_num, length(batch_out_set)) # spec_dim x batch_size

    @assert all(x -> typeof(x) == typeof(batch_out_set[1]), batch_out_set) "All out set must be the same type (Polytope or PolytopeComplement)"
    is_complement = batch_out_set[1] isa Complement

    for (i,o) in enumerate(batch_out_set)
        # A, b = get_linear_spec(o)
        A, b = tosimplehrep(o isa Complement ? Complement(o) : o)
        out_spec_A[1:length(b), 1:size(A,2), i] = A
        out_spec_b[1:length(b), i] = b
        # out_spec_A[1:length(cons), 1:size(A,2), i] = is_complement ? .-A : A
        # out_spec_b[1:length(cons), i] = is_complement ? .-b : b
    end
    # println("out_spec_A")
    # display(out_spec_A)
    # println("out_spec_b")
    # display(out_spec_b)
    return LinearSpec(out_spec_A, out_spec_b, is_complement)
end

"""
    classification_spec(n::Int64, target::Int64)

Generates an output specification constructed with a convex polyhedron, 
`HPolyhedron`, for classification tasks. Given `n`-number of labels with 
`target` as the correct label, the resulting polyhedron is the finite 
intersection of halfspaces:

``
P = \\bigcap_{i=1}^n H_i
``

where ``H_i = \\{x : a_i^T x \\leq 0 \\}, \\; i\\in\\{1:n\\}`` is a halfspace, 
``a_i`` is a row vector where the `n`-th element is 1.0, the `target`-th 
element is -1.0, and the rest are 0's.

## Arguments
- `n` (`Int64`): Number of labels.
- `target` (`Int64`): Target label.

## Returns
- `HPolyhedron` specified as above such that the output specification captures 
    the target label.
"""
function classification_spec(n::Int64, target::Int64)
    A = Matrix{Float64}(I, n, n)
    A[:, target] .= -1
    A = [A[1:target-1, :]; A[target+1:end, :]]
    b = zeros(n-1)
    return HPolyhedron(A, b)
end

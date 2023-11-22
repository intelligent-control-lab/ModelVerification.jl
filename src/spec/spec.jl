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
- `imgs` (`AbstractArray`): list of images in `AbstractArray`. 
    Image is represented as a matrix of h x w x c.
"""
struct ImageConvexHull <: Spec
    # spec: A x - b <= 0 is the safe set or unsafe set
    imgs::AbstractArray # list of images (h,w,c)
end

"""
    LinearSpec <: Spec

Safety specification defined as the set ``\\{ x: x = A x - b ≤ 0 \\}``.

## Fields
- `A` (`AbstractArray{Float64, 3}`): normal dierction of size `spec_dim x out_dim x batch_size`.
- `b` (`AbstractArray{Float64, 2}`): constraints of size `spec_dim x batch_size`.
- `is_complement` (`Bool`): boolean flag for whether this specification is a complement or not.
"""
struct LinearSpec <: Spec 
    # spec: A x - b <= 0 is the safe set or unsafe set
    A::AbstractArray{Float64, 3} # spec_dim x out_dim x batch_size
    b::AbstractArray{Float64, 2} # spec_dim x batch_size
    is_complement::Bool
end

"""
    InputSpec

Input specificaiton can be of any type supported by `LazySet` or `ImageConvexHull`.
"""
const InputSpec = Union{LazySet, ImageConvexHull}

"""
    OutputSpec

Output specification can be of any type supported by `LazySet` or `LinearSpec`.
"""
const OutputSpec = Union{LazySet, LinearSpec}

"""
    get_size(input_spec::LazySet)

Given a `LazySet`, it determines the size.
"""
get_size(input_spec::LazySet) = size(LazySets.center(input_spec))

"""
    get_size(input_spec::ImageConvexHull)

Given an `ImageConvexHull`, it determines the size.
"""
get_size(input_spec::ImageConvexHull) = size(input_spec.imgs[1])

"""
    get_linear_spec(batch_out_set::AbstractVector)

Retrieves the linear specifications of the batch of output sets and returns
a `LinearSpec` structure. 

## Arguments
- `batch_out_set` (`AbstractVector`): batch of output sets.

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
    classification_spec(n, target)
"""
function classification_spec(n, target)
    A = Matrix{Float64}(I, n, n)
    A[:, target] .= -1
    A = [A[1:target-1, :]; A[target+1:end, :]]
    b = zeros(n-1)
    return HPolyhedron(A, b)
end

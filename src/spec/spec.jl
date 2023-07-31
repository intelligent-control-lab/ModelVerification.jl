
abstract type Spec end

struct ImageConvexHull <: Spec
    # spec: A x - b <= 0 is the safe set or unsafe set
    imgs::AbstractArray # list of images (h,w,c)
end

struct LinearSpec <: Spec 
    # spec: A x - b <= 0 is the safe set or unsafe set
    A::AbstractArray{Float64, 3} # spec_dim x out_dim x batch_size
    b::AbstractArray{Float64, 2} # spec_dim x batch_size
    is_complement::Bool
end

const InputSpec = Union{LazySet, ImageConvexHull}
const OutputSpec = Union{LazySet, LinearSpec}

get_size(input_spec::LazySet) = size(LazySets.center(input_spec))
get_size(input_spec::ImageConvexHull) = size(input_spec.imgs[1])


function get_linear_spec(batch_out_set::AbstractVector)
    max_spec_num = maximum([length(constraints_list(o)) for o in batch_out_set])
    out_spec_A = zeros(max_spec_num, dim(batch_out_set[1]), length(batch_out_set)) # spec_dim x out_dim x batch_size
    out_spec_b = zeros(max_spec_num, length(batch_out_set)) # spec_dim x batch_size

    @assert all(x -> typeof(x) == typeof(batch_out_set[1]), batch_out_set) "All out set must be the same type (Polytope or PolytopeComplement)"
    is_complement = batch_out_set[1] isa Complement

    for (i,o) in enumerate(batch_out_set)
        cons = constraints_list(is_complement ? Complement(o) : o)
        A = permutedims(hcat([Vector(con.a) for con in cons]...))
        b = cat([con.b for con in cons], dims=1)
        out_spec_A[1:length(cons), 1:size(A,2), i] = A
        out_spec_b[1:length(cons), i] = b
        # out_spec_A[1:length(cons), 1:size(A,2), i] = is_complement ? .-A : A
        # out_spec_b[1:length(cons), i] = is_complement ? .-b : b
    end
    # println("out_spec_A")
    # display(out_spec_A)
    # println("out_spec_b")
    # display(out_spec_b)
    return LinearSpec(out_spec_A, out_spec_b, is_complement)
end

function classification_spec(n, target)
    A = Matrix{Float64}(I, n, n)
    A[:, target] .= -1
    A = [A[1:target-1, :]; A[target+1:end, :]]
    b = zeros(n-1)
    return HPolyhedron(A, b)
end
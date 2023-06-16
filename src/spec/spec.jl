
abstract type Spec end

struct LinearSpec <: Spec
    # spec: A x - b < 0
    A::AbstractArray{Float64, 3} # spec_dim x out_dim x batch_size
    b::AbstractArray{Float64, 2} # spec_dim x batch_size
end

function get_linear_spec(batch_out_set::AbstractVector)
    max_spec_num = maximum([length(constraints_list(o)) for o in batch_out_set])
    out_spec_A = zeros(max_spec_num, dim(batch_out_set[1]), length(batch_out_set)) # spec_dim x out_dim x batch_size
    out_spec_b = zeros(max_spec_num, length(batch_out_set)) # spec_dim x batch_size
    for (i,o) in enumerate(batch_out_set)
        cons = constraints_list(o)
        A = permutedims(hcat([Vector(con.a) for con in cons]...))
        out_spec_A[1:length(cons), 1:size(A,2), i] = A
        out_spec_b[1:length(cons), i] = cat([con.b for con in cons], dims=1)
    end
    return LinearSpec(out_spec_A, out_spec_b)
end
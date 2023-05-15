include("/home/verification/ModelVerification.jl/src/operator/solver.jl")

function check_forward(forward_reach::Vector{<:LazySet}, output)
    for poly in forward_reach
        issubset(poly, output) || return ReachabilityResult(:violated, forward_reach)
    end
    return ReachabilityResult(:holds, forward_reach)
end

function check_forward(forward_reach::P, output) where P<:LazySet
    return ReachabilityResult(issubset(forward_reach, output) ? :holds : :violated, [forward_reach])
end

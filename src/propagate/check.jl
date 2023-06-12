
function check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::LazySet)
    # println("--")
    #println("input ", low(input), high(input))
    #println("reach ", low(reach), high(reach))
    x = LazySets.center(input)
    #println("center ", x, " ", model(x))
    #println("result ", ⊆(reach, output) ? :holds : ∈(model(x), output) ? :unknown : :violated)
    ⊆(reach, output) && return ReachabilityResult(:holds, [reach])
    ∈(model(x), output) && return CounterExampleResult(:unknown)
    return CounterExampleResult(:violated, x)
end

function check_inclusion(prop_method::ForwardProp, model, input::LazySet, reach::LazySet, output::Complement)
    # println("checking inclusion")
    x = LazySets.center(input)
    unsafe_output = Complement(output)
    # println("checking empty")
    # println(unsafe_output)
    box_reach = box_approximation(reach)
    # println(box_reach)
    # println(isdisjoint(box_reach, unsafe_output))
    # println("===")
    # println("done isempty")
    # ∈(model(x), unsafe_output)
    # println("done issubset")
    isdisjoint(box_reach, unsafe_output) && return ReachabilityResult(:holds, [reach])
    ∈(model(x), unsafe_output) && return CounterExampleResult(:violated, x)
    return CounterExampleResult(:unknown)
end

function check_inclusion(prop_method::ForwardProp, model, batch_input::AbstractArray, batch_reach::AbstractArray, batch_output::AbstractArray)
    results = [check_inclusion(prop_method, model, batch_input[i], batch_reach[i], batch_output[i]) for i in eachindex(batch_reach)]
    return results
end

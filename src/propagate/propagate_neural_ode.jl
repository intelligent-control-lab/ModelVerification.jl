"""
    propagate(prop_method::ODEProp, model_info, batch_info)

Propagates reachable set with a ODE integrator.

## Arguments
- `prop_method` (`ODEProp`): ODE integration method used for the verification 
    process.
- `model_info`: The neural ODE flux model. It is different from model_info in propagate()
    of other, which is a Model structure containing the general computational graph.
- `batch_info`: Dictionary containing information of each node in the model.

## Returns
- `batch_bound`: Bound of the output node, i.e., the final bound.
- `batch_info`: Same as the input `batch_info`, with additional information on 
    the bound of each node in the model.
"""
function propagate(prop_method::ODEProp, ode_model::Chain, batch_info)
    # input: batch x ... x ...

    function black_box_ode!(du, u, p, t)
        y = ode_model(u)
        for i in 1:length(u)
            du[i] = y[i]
        end
    end
    batch_input = batch_info[:batch_input]
    batch_bound = []
    for input_set in batch_input
        sys = BlackBoxContinuousSystem(black_box_ode!, dim(input_set))
        neural_ode_prob = InitialValueProblem(sys, input_set)
        sol = solve(neural_ode_prob, tspan=(0.0, prop_method.t_span), alg=TMJets21a())
        push!(batch_bound, sol[end])
    end
    return batch_bound, batch_info
end
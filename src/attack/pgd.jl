"""
    FGSM(model, loss, x, y; ϵ = 0.1, clamp_range = (0, 1))

Fast Gradient Sign Method (FGSM) is a method of creating adversarial examples
by pushing the input in the direction of the gradient and bounded by the ε parameter.

This method was proposed by Goodfellow et al. 2014 (https://arxiv.org/abs/1412.6572)

## Arguments:
- `model`: The model to base the attack upon.
- `loss`: The loss function to use. This assumes that the loss function includes
    the predict function, i.e. loss(x, y) = crossentropy(model(x), y).
- `x`: The input to be perturbed by the FGSM algorithm.
- `ϵ`: The amount of perturbation to apply.
"""
function FGSM(model, loss, x; ϵ = 0.1)
    J = Flux.gradient(() -> loss(x), Flux.params([x]))
    x_adv = x + (Float32(ϵ) * sign.(J[x]))
    return x_adv
end


"""
    PGD(model, loss, x, y; ϵ = 10, step_size = 0.1, iters = 100, clamp_range = (0, 1))

Projected Gradient Descent (PGD) is an itrative variant of FGSM with a random
point. For every step the FGSM algorithm moves the input in the direction of
the gradient bounded in the l∞ norm.
(https://arxiv.org/pdf/1706.06083.pdf)

## Arguments:
- `model`: The model to base teh attack upon.
- `loss`: the loss function to use, assuming that it includes the prediction function
    i.e. loss(x, y) = crossentropy(m(x), y)
- `x`: The input to be perturbed.
- `step_size`: The ϵ value in the FGSM step.
- `iters`: The maximum number of iterations to run the algorithm for.
"""
function PGD(model, loss, x, input; step_size = 0.001, iters = 100)
    x_adv = x + cpu(randn(Float32, size(x)...)) * Float32(step_size) # start from the random point
    for iter in 1:iters
        ret = x_adv
        x_adv = FGSM(model, loss, x_adv; ϵ = step_size)
        x_adv = project(x_adv, input)
    end
    return x_adv
end

"""
    APGD(model, loss, x, y; ϵ = 10, step_size = 0.1, iters = 100, clamp_range = (0, 1))

Auto Projected Gradient Descent (APGD)
(https://arxiv.org/pdf/2003.01690.pdf)

## Arguments:
- `model`: The model to base teh attack upon.
- `loss`: the loss function to use, assuming that it includes the prediction function
    i.e. loss(x, y) = crossentropy(m(x), y)
- `x`: The input to be perturbed.
- `step_size`: The ϵ value in the FGSM step.
- `rho`: PGD success rate threshold to reduce the step size.
- `a`: momentum.
- `iters`: The maximum number of iterations to run the algorithm for.
"""
function APGD(model, loss, x, input; step_size = 0.001, rho=0.75, a=0.75, iters = 20)
    x_adv = x + cpu(randn(Float32, size(x)...)) * Float32(step_size) # start from the random point
    x_max = x
    x_last = x
    f_max = loss(x)
    f_max_last = f_max
    p_last, p = 0, 0.22
    wj_last, wj = 0, Int(ceil(p * iters))
    success_cnt = 0
    step_size_last = step_size
    for iter in 1:iters
        z_adv = project(FGSM(model, loss, x_adv; ϵ = step_size), input)
        x_last, x_adv = x_adv, project(x_adv + a * (z_adv - x_adv) + (1-a) * (x_adv - x_last), input)
        if loss(x_adv) > f_max
            x_max, f_max = x_adv,loss(x_adv)
            success_cnt += 1
        end
        if iter == wj
            cond1 = success_cnt < rho * (wj - wj_last)
            cond2 = (step_size_last == step_size) & (f_max == f_max_last)
            if cond1 || cond2
                step_size, x_adv = step_size/2, x_max
            end
            f_max_last = f_max
            step_size_last = step_size
            p_last, p = p, p + max(p - p_last - 0.03, 0.06)
            wj_last, wj = wj, Int(ceil(p * iters))
            success_cnt = 0
        end
    end
    return x_max
end

"""
    project(p, rect::Hyperrectangle)
"""
function project(p, rect::Hyperrectangle)
    p = clamp.(p, low(rect), high(rect))
end

"""
    project(p, polytope::LazySet)
"""
function project(p, polytope::LazySet)
    A, b = tosimplehrep(polytope)
    all(A*p < b) && return p
    model = Model(Ipopt.Optimizer)
    # Extract the constraints from the polytope
    A, b = tosimplehrep(polytope)
    @variable(model, x[1:length(p)])
    @objective(model, Min, sum((x[i]-p[i])^2 for i in 1:length(p)))
    # Add the polytope constraints
    for i in 1:size(A, 1)
        @constraint(model, A[i, :]'x <= b[i])
    end
    optimize!(model)
    return value.(x)
end

"""
    attack(model, input, output; restart=100)
"""
function attack(model, input, output; restart=100)
    is_complement = output isa Complement
    Ay, by = tosimplehrep(is_complement ? Complement(output) : output) |> cpu
    # for non-complement,   output should satisfy maximum(A y - b) < 0,  therefore attack maximizes maximum(Ay-b)
    # for complement,       output should satisfy maximum(A y - b) > 0,  therefore attack minimizes maximum(Ay-b), i.e. maximizes -maximum(Ay-b)
    sgn = (is_complement ? -1 : 1)  |> cpu
    loss = (x) -> sgn * maximum(Ay * model(x) - by)
    ϵ = maximum(high(input) - low(input))
    for i in 1:restart
        x = sample(input) |> cpu
        # x_adv = PGD(model |> cpu, loss, x, input; step_size=ϵ/2)
        # println("original loss:", loss(x))
        # println("PGD loss:", loss(x_adv)) 
        x_adv = APGD(model |> cpu, loss, x, input; step_size=ϵ)
        # println("APGD loss:", loss(x_adv))
        # loss(x_adv) > 0 && println("attack success at iter:",i)
        loss(x_adv) > 0 && return CounterExampleResult(:violated, x_adv)
    end
    return BasicResult(:unknown)
end

"""
    attack(problem; restart=100)
"""
function attack(problem; restart=100)
    is_complement = problem.output isa Complement
    Ay, by = tosimplehrep(is_complement ? Complement(problem.output) : problem.output) |> cpu
    # for non-complement,   output should satisfy maximum(A y - b) < 0,  therefore attack maximizes maximum(Ay-b)
    # for complement,       output should satisfy maximum(A y - b) > 0,  therefore attack minimizes maximum(Ay-b), i.e. maximizes -maximum(Ay-b)
    sgn = (is_complement ? -1 : 1)  |> cpu
    loss = (x) -> sgn * maximum(Ay * problem.Flux_model(x) - by)
    ϵ = maximum(high(problem.input) - low(problem.input))
    for i in 1:restart
        x = sample(problem.input) |> cpu
        # x_adv = PGD(problem.Flux_model |> cpu, loss, x, problem.input; step_size=ϵ/2)
        # println("original loss:", loss(x))
        # println("PGD loss:", loss(x_adv))
        x_adv = APGD(problem.Flux_model |> cpu, loss, x, problem.input; step_size=ϵ)
        # println("APGD loss:", loss(x_adv))
        loss(x_adv) > 0 && println("attack success:",i)
        loss(x_adv) > 0 && return CounterExampleResult(:violated, x_adv)
    end
    return BasicResult(:unknown)
end

"""
    project(x, dir::AbstractVector, set::LazySet)
"""
function project(x, dir::AbstractVector, set::LazySet)
    return (x + dir) ∈ set ? (x + dir) : x
end

"""
    project(x, dir::AbstractVector, set::Hyperrectangle)
"""
function project(x, dir::AbstractVector, set::Hyperrectangle)
    c = LazySets.center(set)
    r = radius_hyperrectangle(set)
    return c - r * sign.(dir)
end

# """
#     project(x, dir::AbstractVector, set::LinearSpec)
# """
# function project(x, dir::AbstractVector, set::LinearSpec)
    
# end
function check_iter(
    current_iter::Int64;
    max_iter::Int64 = 100,
    verbosity::Int64 = 2
    )
    if verbosity > 1
        println("Iteration: #", current_iter)
    end
    if current_iter >= max_iter
    println(
            "max_iter reached after ",
            Int(current_iter),
            " iterations",
        )
        return true
    else
        return false
    end
end

function check_time(
    start_time::Float64;
    max_time::Int64 = 1000,
    verbosity::Int64 = 2
    )
    current_time = time()-start_time
    if current_time>= max_time
    println(
            "max_time reached after ",
            current_time,
            " seconds",
        )
        return true
    else
        return false
    end
end

function check_f_tol_rel!(
    examinees::Vector{<:AbstractExaminee},
    old_likelihood::Float64;
    f_tol_rel::Float64 = 0.000001,
    verbosity::Int64 = 2
    )
    new_likelihood = likelihood(examinees)
    f_rel = abs((new_likelihood - old_likelihood) / old_likelihood)
    if verbosity > 1
        println("Likelihood: ", new_likelihood)
    end
    if f_rel < f_tol_rel
        println(
            "f_tol_rel reached"
        )
        old_likelihood = copy(new_likelihood)
        return true
    else
        old_likelihood = copy(new_likelihood)
        return false
    end
end

function check_f_tol_rel!(
    latents::Vector{<:AbstractLatent},
    old_likelihood::Float64;
    f_tol_rel::Float64 = 0.000001,
    verbosity::Int64 = 2
    )
    new_likelihood = _likelihood(latents)
    f_rel = abs((new_likelihood - old_likelihood) / old_likelihood)
    if verbosity > 1
        println("Likelihood: ", new_likelihood)
    end
    if f_rel < f_tol_rel
        println(
            "f_tol_rel reached"
        )
        old_likelihood = copy(new_likelihood)
        return true
    else
        old_likelihood = copy(new_likelihood)
        return false
    end
end
   

function check_f_tol_rel!(
    new_likelihood::Float64,
    old_likelihood::Float64;
    f_tol_rel::Float64 = 0.01,
    verbosity::Int64 = 2
    )
    f_rel = abs((new_likelihood - old_likelihood) / old_likelihood)
    if verbosity > 1
        println("Likelihood: ", new_likelihood)
    end
    if f_rel < f_tol_rel
        println(
            "f_tol_rel reached"
        )
        old_likelihood = copy(new_likelihood)
        return true
    else
        old_likelihood = copy(new_likelihood)
        return false
    end
end

function check_x_tol_rel!(
    new_pars::Matrix{Float64},
    old_pars::Matrix{Float64};
    x_tol_rel::Float64 = 0.0001,
    verbosity::Int64 = 2
)
    delta_pars =  maximum(abs.((new_pars - old_pars) ./ old_pars))
    if verbosity > 1
        println("x_rel max: ", delta_pars)
    end
    if delta_pars <= x_tol_rel
        println(
           "X ToL reached"
        )
        old_pars .= new_pars
        return true
    else
        old_pars .= new_pars
        return false
    end
end

function check_x_tol_rel!(
    parameters::Vector{<:AbstractParameters},
    old_pars::Matrix{Float64};
    x_tol_rel::Float64 = 0.0001,
    verbosity::Int64 = 2
)
    new_pars = hcat(_get_parameters_vals.(parameters)...)
    delta_pars =  maximum(abs.((new_pars - old_pars) ./ old_pars))
    if verbosity > 1
        println("x_rel max: ", delta_pars)
    end
    if delta_pars <= x_tol_rel
        println(
           "X ToL reached"
        )
        old_pars .= new_pars
        return true
    else
        old_pars .= new_pars
        return false
    end
end

function check_x_tol_rel!(
    items::Vector{<:AbstractItem},
    old_pars::Matrix{Float64};
    x_tol_rel::Float64 = 0.0001,
    verbosity::Int64 = 2
)
    new_pars = get_parameters_vals(items)
    delta_pars =  maximum(abs.((new_pars - old_pars) ./ old_pars))
    if verbosity > 1
        println("x_rel max: ", delta_pars)
    end
    if delta_pars <= x_tol_rel
        println(
           "X ToL reached"
        )
        old_pars .= new_pars
        return true
    else
        old_pars .= new_pars
        return false
    end
end

function check_x_tol_rel!(
    latents::Vector{<:AbstractLatent},
    old_latents::Matrix{Float64};
    x_tol_rel::Float64 = 0.0001,
    verbosity::Int64 = 2
)
    new_latents = hcat(_get_latents.(latents)...)
    delta_latents =  maximum(abs.((new_latents - old_latents) ./ old_latents))
    if verbosity > 1
        println("x_rel max: ", delta_pars)
    end
    if delta_latents <= x_tol_rel
        println(
           "X ToL reached"
        )
        old_latents .= new_latents
        return true
    else
        old_latents .= new_latents
        return false
    end
end
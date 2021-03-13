function check_iter(
    current_iter::Int64;
    max_iter::Int64 = 100
    )
    println("Iteration: #", current_iter)
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
    max_time::Int64 = 1000
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
    f_tol_rel::Float64 = 0.000001
    )
    new_likelihood = likelihood(examinees)
    f_rel = abs((new_likelihood - old_likelihood) / old_likelihood)
    println("Likelihood: ", new_likelihood)
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
    f_tol_rel::Float64 = 0.000001
    )
    new_likelihood = _likelihood(latents)
    f_rel = abs((new_likelihood - old_likelihood) / old_likelihood)
    println("Likelihood: ", new_likelihood)
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
    f_tol_rel::Float64 = 0.000001
    )
    f_rel = abs((new_likelihood - old_likelihood) / old_likelihood)
    println("Likelihood: ", new_likelihood)
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
    items::Vector{<:AbstractItem},
    old_pars::Matrix{Float64};
    x_tol_rel::Float64 = 0.0001
)
    new_pars = get_parameters_vals(items)
    delta_pars =  maximum(abs.((new_pars - old_pars) ./ old_pars))
    println("x_rel max: ", delta_pars)
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
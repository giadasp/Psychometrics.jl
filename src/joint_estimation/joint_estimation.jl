include("polyagamma/polyagamma_mcmc.jl")
include("mmle/je_mmle.jl")

function joint_estimate!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    method = "mmle",
    quick = false,
    kwargs...
    )
    if method == "pg"
        if quick
            return joint_estimate_pg_quick!(items, examinees, responses; kwargs...)
        else
            return joint_estimate_pg!(items, examinees, responses; kwargs...)
        end
    elseif method =="mmle"
        if quick
            return joint_estimate_mmle_quick!(items, examinees, responses; kwargs...)
        else
            return joint_estimate_mmle!(items, examinees, responses; kwargs...)
        end
    end
end




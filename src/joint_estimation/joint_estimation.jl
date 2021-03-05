include("polyagamma/polyagamma_mcmc.jl")
include("mmle/mmle.jl")

function joint_estimate!(
    item::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    method = "mmle",
    kwargs...
    )
    if method == "pg"
        return joint_estimate_pg!(item, examinees, responses; kwargs...)
    elseif method =="mmle"
        return joint_estimate_mmle!(item, examinees, responses; kwargs...)
    end
end


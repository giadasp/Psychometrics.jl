include("polyagamma/polyagamma_mcmc.jl")
include("mmle/mmle.jl")

function assess_examinee!(
    examinee::AbstractExaminee,
    items::Vector{<:AbstractItem},
    responses::Vector{Response};
    method = "mmle",
    kwargs...
    )
    if method == "pg"
        return assess_examinee_pg!(item, examinees, responses; kwargs...)
    elseif method =="mmle"
        return assess_examinee_mmle!(item, examinees, responses; kwargs...)
    end
end


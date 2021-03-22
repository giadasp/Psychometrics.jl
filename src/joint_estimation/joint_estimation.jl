include("polyagamma/je_polyagamma.jl")
include("mmle/je_mmle.jl")

function joint_estimate!(
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    method = "mmle",
    kwargs...
    )
    if method == "pg"
        return joint_estimate_pg!(items, examinees, responses; kwargs...)
    elseif method =="mmle"
        return joint_estimate_mmle!(items, examinees, responses; kwargs...)
    end
end




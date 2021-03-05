
include("polyagamma/polyagamma_mcmc.jl")
include("mmle/mmle.jl")

function calibrate_item!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Response};
    method = "mmle",
    kwargs...
    )
    if method == "pg"
        return calibrate_item_pg!(item, examinees, responses; kwargs...)
    elseif method =="mmle"
        return calibrate_item_mmle!(item, examinees, responses; kwargs...)
    end
end


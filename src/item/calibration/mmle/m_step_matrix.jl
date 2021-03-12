function _calibrate_item_mmle!(
    parameters::Parameters2PL,
    examinees::Vector{<:AbstractExaminee}, #only those who answered to item 
    responses::Vector{Union{Missing, Float64}},
    opt::NLopt.Opt
)
    opt.lower_bounds = [parameters.bounds_b[1], parameters.bounds_a[1]]
    opt.upper_bounds = [parameters.bounds_b[2], parameters.bounds_a[2]]
    sumpk_i = mapreduce( e -> e.latent.posterior.p, +, examinees)
    if sum(responses)>0
        r1_i = mapreduce(e -> e.latent.posterior.p, +, examinees[responses .> 0])
    else
        r1_i = [0.0]
    end
    pars_i = max_i(examinees[1].latent.posterior.support, sumpk_i, r1_i, [parameters.b, parameters.a], opt)
    parameters.a = pars_i[2]
    parameters.b = pars_i[1]
    return nothing
end

function calibrate_item_mmle!(
    item::AbstractItem,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{Union{Missing, Float64}},
    opt::NLopt.Opt
)
    _calibrate_item_mmle!(item.parameters,
        examinees,
        responses,
        opt
    )
    return nothing
end

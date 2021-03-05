
function _generate_w(parameters::Parameters2PL, latent::Latent1D)
    return Distributions.rand(PolyaGamma(1, parameters.a * (latent.val - parameters.b)))
end

function generate_w(item::AbstractItem, examinees::Vector{<:AbstractExaminee})
    return map(
        e -> PolyaGammaSample(item.idx, e.idx, _generate_w(item.parameters, e.latent)),
        examinees,
    )
end

function generate_w(
    items::Vector{<:AbstractItem},
    examinees_i::Vector{Vector{Examinee}},
)
    return mapreduce(
        i -> map(
            e -> PolyaGammaSample(i.idx, e.idx, _generate_w(i.parameters, e.latent)),
            examinees_i[i.idx],
        ),
        vcat,
        items,
    )
end


function _generate_w(latent::Latent1D, parameters::Parameters2PL)
    return Distributions.rand(PolyaGamma(1, parameters.a * (latent.val - parameters.b)))
end

function generate_w(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
    return map(
        i -> PolyaGammaSample(
            i.idx,
            examinee.idx,
            _generate_w(i.parameters, examinee.latent)
        ),
        items,
    )
end

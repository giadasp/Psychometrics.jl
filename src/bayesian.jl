mutable struct PolyaGammaSample
    i_idx::Int64
    e_idx::Int64
    val::Float64
    PolyaGammaSample(i_idx, e_idx, val) = new(i_idx, e_idx, val)
end
function update_posterior!(
    item::Item2PL,
    examinees::Vector{<:AbstractExaminee},
    responses::Vector{<:AbstractResponse}, #only responses of item
    W::Vector{Float64},
)

    #W = map( e -> Distributions.rand(PolyaGammaDevRoye1Sampler(1.0, item.parameters.a *(e.latent.val - item.parameters.b))), examinees)
    sigma2 = mapreduce(
        (e, w) -> [(e.latent.val - item.parameters.b)^2 * w, (item.parameters.a)^2 * w],
        +,
        examinees,
        W,
    )
    sigma2 = 1 ./ (sigma2 + (1 ./ Distributions.var(item.parameters.prior)))
    mu = mapreduce(
        (e, w) -> [
            (e.latent.val - item.parameters.b) *
            (get_responses_by_examinee_id(e.id, responses)[1].val - 0.5),
            -item.parameters.a * (
                (get_responses_by_examinee_id(e.id, responses)[1].val - 0.5) -
                (item.parameters.a * e.latent.val * w)
            ),
        ],
        +,
        examinees,
        W,
    )
    mu =
        sigma2 .* (
            mu + (
                Distributions.mean.(item.parameters.prior.v) ./
                Distributions.var.(item.parameters.prior.v)
            )
        )

    item.parameters.posterior = Distributions.Product([
        Distributions.TruncatedNormal(mu[1], sqrt(sigma2[1]), 0.0, 10.0),
        Distributions.Normal(mu[2], sqrt(sigma2[2])),
    ])
end


function update_posterior!(
    examinee::Examinee1D,
    items_e::Vector{<:AbstractItem},
    responses_e::Vector{<:AbstractResponse}, #only responses of examinee
    W::Vector{Float64},
)
    #W = map( i -> Distributions.rand(PolyaGammaDevRoye1Sampler(1, i.parameters.a *(examinee.latent.val - i.parameters.b))), items_e)
    sigma2 = mapreduce((i, w) -> (i.parameters.a^2) * w, +, items_e, W)
    sigma2 = 1 / (sigma2 + (1 / Distributions.var(examinee.latent.prior)))
    mu =
        sigma2 * (
            mapreduce(
                (i, w) ->
                    i.parameters.a * (
                        i.parameters.a * i.parameters.b * w +
                        (get_responses_by_item_id(i.id, responses_e)[1].val - 0.5)
                    ),
                +,
                items_e,
                W,
            ) + (examinee.latent.prior.μ / Distributions.var(examinee.latent.prior))
        )
    examinee.latent.posterior = Distributions.Normal(mu, sqrt(sigma2))
end

function generate_w(items::Vector{<:AbstractItem}, examinees_i::Vector{Vector{Examinee1D}})
    return mapreduce(
        i -> map(
            e -> PolyaGammaSample(
                e.idx,
                i.idx,
                Distributions.rand(Psychometrics.PolyaGammaSPSampler(
                    1,
                    i.parameters.a * (e.latent.val - i.parameters.b),
                )),
            ),
            examinees_i[i.idx],
        ),
        vcat,
        items,
    )
end

#extract random values

function set_value_from_posterior!(item::Item2PL)
    vals = Distributions.rand(item.parameters.posterior)
    item.parameters.a = vals[1]
    item.parameters.b = vals[2]
end

function set_value_from_posterior!(examinee::Examinee1D)
    examinee.latent.val = Distributions.rand(examinee.latent.posterior)
end

function set_value_from_chain!(item::Item2PL)
    item.parameters.a = item.parameters.chain[end][1]
    item.parameters.b = item.parameters.chain[end][2]
end

function set_value_from_chain!(examinee::Examinee1D)
    examinee.latent.val = examinee.latent.chain[end]
end

function chain_append!(item::Item2PL)
    push!(item.parameters.chain, Distributions.rand(item.parameters.posterior))
end

function chain_append!(examinee::Examinee1D)
    push!(examinee.latent.chain, Distributions.rand(examinee.latent.posterior))
end

#Jiang and Templin

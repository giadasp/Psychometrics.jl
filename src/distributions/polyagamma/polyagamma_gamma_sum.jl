struct PolyaGammaGammaSumSampler <: Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}
 h::Float64
 z::Float64
end


const bvec = map(x -> (x - 0.5)^2 * pi^2 * 4, 1:_TERMS)

## draw sum of gammas
function Distributions.rand(rng::Distributions.AbstractRNG, s::PolyaGammaGammaSumSampler)
    g = Distributions.rand(rng, Distributions.Gamma(s.h), _TERMS)
    # x = @distributed (+) for k in bvec
    # 2 * Distributions.rand(rng, g) / (s.z^2 + k)
    # end
    # return x
    # b =  mapreduce(k -> 1 / (k + s.z^2), +, bvec2)
    #return Distributions.rand(rng,Gamma(2 * s.h * b))#
    # return (2 * sum( map(bvec, g) do k, g_i
    #          g_i / (k + s.z^2)
    #         end )
    return 2 * mapreduce((k, g_i) -> Random.rand(rng, g) / (k + s.z^2), +, bvec, g)
end
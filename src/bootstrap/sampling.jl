function parametric_sample(
    N::Int64,
    sample_fraction::Float64,
    bins::Vector{Int64},
    dist::Distributions.DiscreteNonParametric
    )
    W = dist.p
    K = size(W, 1)
    n_sample = rand(
        Distributions.DiscreteNonParametric(collect(1:K), W),
        Int(floor(sample_fraction * N)),
    )
    #check if each n_sample exists in bins, in case sample another one
    for n = 1:size(n_sample, 1)
        ns = copy(n_sample[n])
        while size(findall(bins .== ns), 1) == 0
            # if n_sample[n] > (K / 2)
            #     ns -= 1
            # else
            #     ns += 1
            # end
            ns = rand(Distributions.DiscreteNonParametric(collect(1:K), W),1)
        end
        n_sample[n] = copy(ns[1])
    end
    return n_sample::Vector{Int64}
end

function non_parametric_sample(
    N::Int64,
    sample_fraction::Float64
)
    return Distributions.sample(
        collect(1:N),
        Int(ceil(sample_fraction * N)),
        replace = true
    )
end
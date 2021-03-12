"""
    get_latents(examinees::Vector{<:AbstractExaminee})

    #Description 

Returns the latent objects of a vector of examinees as a vector of subtypes of `AbstractLatent`.
"""
function get_latents(examinees::Vector{<:AbstractExaminee})
    map( e -> e.latent, examinees)::Vector{<:AbstractLatent}
end


"""
    get_latents_vals(examinees::Vector{<:AbstractExaminee})

    #Description 

Returns a matrix with latent values displayed by row.
"""
function get_latents_vals(examinees::Vector{<:AbstractExaminee})
    ret = Vector{Vector{Float64}}(undef, size(examinees, 1))
    max_length = size(get_latents_vals(examinees[1]), 1)
    e_2 = 0
    #compute maximum latent dimension
    for e in examinees
        e_2 += 1
        local latents = get_latents_vals(e)
        if size(latents, 1) == 1
            ret[e_2] = [latents]
        else
            ret[e_2] = latents
        end
        #create vectors
        if size(latents, 1) > max_length
            max_length = size(latents, 1)
            for e_3 = 1:(e_2-1)
                local length_i = size(ret[e_3], 1)
                if length_i < max_length
                    ret[e_3] .= vcat(ret[e_3], zeros(Float64, max_length - length_i))
                end
            end
        end
    end
    #hcat vectors
    return reduce(hcat, ret)::Matrix{Float64}
end

"""
    set_prior!(
        examinees::Vector{<:AbstractExaminee},
        prior::Union{Distributions.DiscreteUnivariateDistribution, Distributions.ContinuousUnivariateDistribution}
    )
"""
function  set_prior!(
    examinees::Vector{<:AbstractExaminee},
    prior::Union{Distributions.DiscreteUnivariateDistribution, Distributions.ContinuousUnivariateDistribution}
)
    map( e -> _set_prior!(e.latent, prior), examinees)
    return nothing
end
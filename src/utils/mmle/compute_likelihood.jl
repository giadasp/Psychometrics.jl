function likelihood(
    examinees::Vector{<:AbstractExaminee}
    )
    return mapreduce( e -> _log_c(e.latent.likelihood), +, examinees)::Float64
end

function _likelihood(
    latents::Vector{<:AbstractLatent}
    )
    return mapreduce( l -> _log_c(l.likelihood), +, latents)::Float64
end


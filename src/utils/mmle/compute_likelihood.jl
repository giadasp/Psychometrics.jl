function likelihood(
    examinees::Vector{<:AbstractExaminee}
    )
    return mapreduce( e -> _log_c(e.latent.likelihood), +, examinees)::Float64
end


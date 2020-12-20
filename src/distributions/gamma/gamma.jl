function Distributions.minimum(d::Distributions.Gamma{T}) where {T<:Real}
    return 0.0
end
function Distributions.maximum(d::Distributions.Gamma{T}) where {T<:Real}
    return Inf
end

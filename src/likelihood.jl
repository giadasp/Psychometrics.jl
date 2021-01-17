"""
log_likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary,
)

# Description

It computes the log likelihood for a matrix of latent values, item parameters, responses and design. 
"""
function log_likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary,
)
    p = __probability(latent_val, parameters)
    return response_val * log(p) + (1 - response_val) * log(1 - p)
end

"""
_likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary,
)

#Description

It computes the log likelihood for a latent value and item parameters `parameters` with answer `response_val`.
"""
function _likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary,
)
    p = __probability(latent_val, parameters)
    return p^response_val * (1 - p)^(1 - response_val)
end

"""
_log_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
)

# Description

It computes the log likelihood for a 1-dimensional latent variable and item parameters `parameters` with answer `response_val`.
"""
function _log_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
)
    p = _probability(latent, parameters)
    return response_val * log(p) + (1 - response_val) * log(1 - p)
end


"""
_log_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)

# Description

It computes the log likelihood for a 1-dimensional latent variable and item parameters `parameters` with answer `response_val`. 
It updates also the gradient vectors.
"""
function _log_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)

    if size(g_item, 1) > 0 || size(g_latent, 1) > 0
        p = _probability(latent, parameters, g_item, g_latent)

        if size(g_item, 1) > 0
            g_item .=
                (response_val / p .* g_item) - ((1 - response_val) / (1 - p) .* g_item)
        end

        if size(g_latent, 1) > 0
            g_latent .=
                (response_val / p .* g_latent) - ((1 - response_val) / (1 - p) .* g_latent)
        end
    end

    return _log_likelihood(response_val, latent, parameters)
end

"""
log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)

# Description

It computes the log likelihood for a `response`. 
"""
function log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)
    _log_likelihood(response.val, examinee.latent, item.parameters)
end

"""
log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)

# Description

It computes the log likelihood for a response `response`. 
It updates also the gradient vectors.
"""
function log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
    _log_likelihood(response.val, examinee.latent, item.parameters, g_item, g_latent)
end

"""
log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)

# Description

It computes the log likelihood for a vector of responses `responses`. 
"""
function log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
    map(r -> log_likelihood(r, examinees[r.examinee_idx], items[r.item_idx]), responses)
end

"""
log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)

# Description

It computes the log likelihood for a vector of responses `responses`. 
It updates also the gradient vectors.
"""
function log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
    map(
        r -> log_likelihood(
            r,
            examinees[r.examinee_idx],
            items[r.item_idx],
            g_item,
            g_latent,
        ),
        responses,
    )
end



"""
_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
)

# Description

It computes the likelihood for a 1-dimensional latent variable and item parameters `parameters` with answer `response_val`.
"""
function _likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
)
    return _likelihood(response_val, latent.val, parameters)
end

"""
likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)

# Description

It computes the log likelihood for a `response`. 
"""
function likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)
    return _likelihood(response.val, examinee.latent.val, item.parameters)
end

"""
likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)

# Description

It computes the log likelihood for a vector of responses `responses`. 
"""
function likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
    map(r -> likelihood(r, examinees[r.examinee_idx], items[r.item_idx]), responses)
end

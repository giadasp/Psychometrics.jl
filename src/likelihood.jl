"""
```julia 
log_likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary,
)
```

# Description

It computes the log likelihood for a latent value and item parameters `parameters` with answer `response_val`.
"""
function log_likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary,
)
    p = __probability(latent_val, parameters)
    if (response_val > 0)
        return _log_c(p) * weight::Float64
    else
        return _log_c(1 - p) * weight::Float64
    end
end

"""
```julia 
__likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary;
    weight::Float64 = 1.0
)
```

#Description

It computes the likelihood for a latent value and item parameters `parameters` with answer value `response_val`.
Optionally weights the result by setting `weight`, that is equal to 1.0 by default.
"""
function __likelihood(
    response_val::Float64,
    latent_val::Float64,
    parameters::AbstractParametersBinary;
    weight::Float64 = 1.0
)
    p = __probability(latent_val, parameters)
    if (response_val > 0)
        return p * weight::Float64
    else
        return (1 - p) * weight::Float64
    end
end

function _likelihood(
    response_val::Float64,
    latent_val::Float64,
    item::AbstractItem;
    weight::Float64 = 1.0
)
    return __likelihood(response_val, latent_val, item.parameters; weight = weight)
end


"""
```julia 
_likelihood(
    response::AbstractResponse,
    latent_val::Float64,
    item::AbstractItem;
    weight::Float64 = 1.0
)
```

#Description

It computes the likelihood for a latent value and item `item` with answer `response`.
Optionally weights the result by setting `weight`, that is equal to 1.0 by default.
"""
function _likelihood(
    response::AbstractResponse,
    latent_val::Float64,
    item::AbstractItem;
    weight::Float64 = 1.0
)
    return __likelihood(response.val, latent_val, item.parameters; weight = weight)
end


"""
```julia 
_log_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
)
```

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
```julia 
_log_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
```

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
```julia 
log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)
```

# Description

It computes the log likelihood for a triplet `response`, `examinee`, `item`. 
"""
function log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)
    _log_likelihood(response.val, examinee.latent, item.parameters)
end

"""
```julia 
log_likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
```

# Description

It computes the log likelihood for a triplet `response`, `examinee`, `item`. 
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
```julia 
log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
```

# Description

It computes the log likelihood for a vector of responses `responses`, and provided `examinees` and `items` vectors. 
"""
function log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
    map(r -> log_likelihood(r, examinees[r.examinee_idx], items[r.item_idx]), responses)
end

"""
```julia 
log_likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
    g_item::Vector{Float64},
    g_latent::Vector{Float64},
)
```

# Description

It computes the log likelihood for a vector of responses `responses`, and provided `examinees` and `items` vectors. 
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
```julia 
_likelihood(
    response_val::Float64,
    latent::Latent1D,
    parameters::AbstractParametersBinary,
)
```

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
```julia 
likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)
```

# Description

It computes the likelihood for a triplet `response`, `examinee`, `item`. 
"""
function likelihood(
    response::AbstractResponse,
    examinee::AbstractExaminee,
    item::AbstractItem,
)
    return _likelihood(response.val, examinee.latent.val, item.parameters)
end

"""
```julia 
likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
```

# Description

It computes the likelihood for a vector of responses `responses`, and provided `examinees` and `items` vectors. 
"""
function likelihood(
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
    map(r -> likelihood(r, examinees[r.examinee_idx], items[r.item_idx]), responses)
end
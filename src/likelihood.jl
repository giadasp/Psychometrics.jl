"""
    log_likelihood(latent::Latent1D, parameters::AbstractParameters, response_val::Float64)

It computes the log likelihood for a 1-dimensional latent variable and item parameters `parameters` with answer `response_val`.
"""
function log_likelihood(latent::Latent1D, parameters::AbstractParameters, response_val::Float64)
    p =  probability(latent, parameters)    
    return response_val * log(p) + (1 - response_val) * log(1 - p)
end

"""
    log_likelihood(latent::Latent1D, parameters::AbstractParameters, response_val::Float64, g_item::Vector{Float64}, g_latent::Vector{Float64})

It computes the log likelihood for a 1-dimensional latent variable and item parameters `parameters` with answer `response_val`. 
It updates also the gradient vectors.
"""
function log_likelihood(latent::Latent1D, parameters::AbstractParameters, response_val::Float64, g_item::Vector{Float64}, g_latent::Vector{Float64})
    p =  probability(latent, parameters,  g_item, g_latent)
        
        if size(g_item, 1)>0
            g_item .= (response_val / p .* g_item) - ((1 - response_val) / (1 - p) .* g_item)
        end    
        
        if size(g_latent, 1)>0
            g_latent .= (response_val / p .* g_latent) - ((1 - response_val) / (1 - p) .* g_latent)
        end
        
    
    return log_likelihood(latent, parameters, response_val)
end

"""
    log_likelihood(response::AbstractResponse, g_item::Vector{Float64}, g_latent::Vector{Float64})

It computes the log likelihood for a response `response`. 
"""
function log_likelihood(response::AbstractResponse)
    log_likelihood(response.examinee.latent, response.item.parameters, response.val)
end

"""
    log_likelihood(response::AbstractResponse, g_item::Vector{Float64}, g_latent::Vector{Float64})

It computes the log likelihood for a response `response`. 
It updates also the gradient vectors.
"""
function log_likelihood(response::AbstractResponse, g_item::Vector{Float64}, g_latent::Vector{Float64})
    log_likelihood(response.examinee.latent, response.item.parameters, response.val, g_item, g_latent)
end

"""
    log_likelihood(responses::Vector{<:AbstractResponse}, g_item::Vector{Float64}, g_latent::Vector{Float64})

It computes the log likelihood for a vector of responses `responses`. 
It updates also the gradient vectors.
"""
function log_likelihood(responses::Vector{<:AbstractResponse}, g_item::Vector{Float64}, g_latent::Vector{Float64})
    ret = map(r -> log_likelihood(r, g_item, g_latent), responses) 
    return sum(ret)
end


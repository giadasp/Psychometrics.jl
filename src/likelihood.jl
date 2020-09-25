"""
    log_likelihood(response_val::Float64, latent_val::Float64, parameters::AbstractParameters)

It computes the log likelihood for a latent value and item parameters `parameters` with answer `response_val`.
"""
function log_likelihood(response_val::Float64, latent_val::Float64, parameters::AbstractParameters)
p =  probability(latent_val, parameters)    
return response_val * log(p) + (1 - response_val) * log(1 - p)
end

"""
    likelihood(response_val::Float64, latent_val::Float64, parameters::AbstractParameters)

It computes the log likelihood for a latent value and item parameters `parameters` with answer `response_val`.
"""
function likelihood(response_val::Float64, latent_val::Float64, parameters::AbstractParameters)
p =  probability(latent_val, parameters)    
return p^response_val * (1 - p)^(1 - response_val)
end

"""
    log_likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParameters)

It computes the log likelihood for a 1-dimensional latent variable and item parameters `parameters` with answer `response_val`.
"""
function log_likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParameters)
    p =  probability(latent, parameters)    
    return response_val * log(p) + (1 - response_val) * log(1 - p)
end


"""
    log_likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParameters, g_item::Vector{Float64}, g_latent::Vector{Float64})

It computes the log likelihood for a 1-dimensional latent variable and item parameters `parameters` with answer `response_val`. 
It updates also the gradient vectors.
"""
function log_likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParameters, g_item::Vector{Float64}, g_latent::Vector{Float64})
        
    if size(g_item, 1)>0 || size(g_latent, 1)>0
        p =  probability(latent, parameters,  g_item, g_latent)
        
        if size(g_item, 1)>0
            g_item .= (response_val / p .* g_item) - ((1 - response_val) / (1 - p) .* g_item)
        end    
        
        if size(g_latent, 1)>0
            g_latent .= (response_val / p .* g_latent) - ((1 - response_val) / (1 - p) .* g_latent)
        end
    end
    
    return log_likelihood(response_val, latent, parameters)
end

"""
    log_likelihood(response::AbstractResponse, examinee::AbstractExaminee, item::AbstractItem)

It computes the log likelihood for a `response`. 
"""
function log_likelihood(response::AbstractResponse, examinee::AbstractExaminee, item::AbstractItem)
    log_likelihood(response.val, examinee.latent, item.parameters)
end

"""
    log_likelihood(response::AbstractResponse, examinee::AbstractExaminee, item::AbstractItem, g_item::Vector{Float64}, g_latent::Vector{Float64})

It computes the log likelihood for a response `response`. 
It updates also the gradient vectors.
"""
function log_likelihood(response::AbstractResponse, examinee::AbstractExaminee, item::AbstractItem, g_item::Vector{Float64}, g_latent::Vector{Float64})
    log_likelihood(response.val, examinee.latent, item.parameters, g_item, g_latent)
end

"""
    log_likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem}, g_item::Vector{Float64}, g_latent::Vector{Float64})

It computes the log likelihood for a vector of responses `responses`. 
It updates also the gradient vectors.
"""
function log_likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem}, g_item::Vector{Float64}, g_latent::Vector{Float64})
    ret = map(r -> log_likelihood(r, examinees[r.examinee_idx], items[r.item_idx], g_item, g_latent), responses) 
    return sum(ret)
end

"""
    log_likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem})

It computes the log likelihood for a vector of responses `responses`. 
"""
function log_likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem})
    mapreduce(r -> log_likelihood(r, examinees[r.examinee_idx], items[r.item_idx]), +, responses)
end

"""
    log_likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem}, g_item::Vector{Float64}, g_latent::Vector{Float64})

It computes the log likelihood for a vector of responses `responses`. 
It updates also the gradient vectors.
"""
function log_likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem}, g_item::Vector{Float64}, g_latent::Vector{Float64})
    mapreduce(r -> log_likelihood(r, examinees[r.examinee_idx], items[r.item_idx], g_item, g_latent), +, responses)
end



"""
    likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParameters)

It computes the likelihood for a 1-dimensional latent variable and item parameters `parameters` with answer `response_val`.
"""
function likelihood(response_val::Float64, latent::Latent1D, parameters::AbstractParameters)
    return likelihood(response_val, latent.val, parameters)
end

"""
    likelihood(response::AbstractResponse, examinee::AbstractExaminee, item::AbstractItem)

It computes the log likelihood for a `response`. 
"""
function likelihood(response::AbstractResponse, examinee::AbstractExaminee, item::AbstractItem)
    return likelihood(response.val, examinee.latent.val, item.parameters)
end

"""
    likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem}

It computes the likelihood for a vector of responses `responses`. 
"""
function likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem})
    ret = map(r -> likelihood(r, examinees[r.examinee_idx], items[r.item_idx]), responses) 
    return sum(ret)
end

"""
    likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem})

It computes the log likelihood for a vector of responses `responses`. 
"""
function likelihood(responses::Vector{<:AbstractResponse},  examinees::Dict{Int64,<:AbstractExaminee}, items::Dict{Int64,<:AbstractItem})
    mapreduce(r -> likelihood(r, examinees[r.examinee_idx], items[r.item_idx]), *, responses)
end

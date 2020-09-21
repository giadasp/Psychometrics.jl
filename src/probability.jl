function probability(latent::Latent1D, parameters::Parameters1PL)
    1 / (1 + _exp_c(parameters.b - latent.val))
end

function probability(latent::Latent1D, parameters::Parameters1PL, g_item::Vector{Float64}, g_latent::Vector{Float64})
    p = probability(latent, parameters)
    
    if size(g_item,1)>0
        g_item .= p * (1 - p)
    end
    
    if size(g_latent,1)>0
        g_latent .= - p * (1 - p)
    end
    
    return p
end

function probability(latent::Latent1D, parameters::Parameters2PL)
   1 / (1 + _exp_c( - parameters.a * (latent.val - parameters.b)))
end

function probability(latent::Latent1D, parameters::Parameters2PL, g_item::Vector{Float64}, g_latent::Vector{Float64})
    p = probability(latent, parameters)
    
    if size(g_item,1)>0 || size(g_latent,1)>0
        p1p = p * (1 - p)
        if size(g_item,1)>0
            g_item .= [(latent.val - parameters.b) * p1p ,- parameters.a * p1p ]
        end
        
        if size(g_latent,1)>0
            g_latent .= parameters.a * p1p
        end
    end
    
    return p
end

function probability(latent::Latent1D, parameters::Parameters3PL)
    parameters.c + (1 - parameters.c) * (1 / (1 + _exp_c( - parameters.a * (latent.val - parameters.b ))))
end

function probability(latent::Latent1D, parameters::Parameters3PL,  g_item::Vector{Float64}, g_latent::Vector{Float64})
    p = probability(latent, parameters)
    
    if size(g_item,1)>0 || size(g_latent,1)>0
        q1c = (1 - p) / (1 - parameters.c)
        
        if size(g_item,1)>0
            g_item .= [(latent.val - parameters.b) * q1c * (p - parameters.c), - parameters.a * q1c * (p - parameters.c) , q1c]
        end    
        
        #by Kim's book
        if size(g_latent,1)>0
            g_latent .= parameters.a * (p - parameters.c) * q1c
        end
        
    end
    
    return p
end
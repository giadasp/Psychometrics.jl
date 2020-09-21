
function information_latent(latent::Latent1D, parameters::Parameters2PL)
    p = probability(latent.val, parameters)
    return p * (1 - p) * parameters.a^2
end

function information_latent(latent::Latent1D, parameters::Parameters3PL)
    p = probability(latent.val, parameters)
    return (p - parameters.c)^2 * (1 - p) * parameters.a^2 / (1 - parameters.c)^2 / p
end

function  expected_information_item(latent::Latent1D, parameters::Parameters1PL)
    p = probability(latent.val, parameters)
    return (latent.val - parameters.b)^2 * p * (1 - p)
end

function  expected_information_item(latent::Latent1D, parameters::Parameters2PL)
    p = probability(latent.val, parameters)
    i_aa =                       (1 - p) * p *                         (latent.val - parameters.b)^2 
    i_ab = - parameters.a * (1 - p) * p *                         (latent.val - parameters.b)  
    i_bb = parameters.a^2 * (1 - p) * p *
    return [i_aa  i_ab; i_ab  i_bb]
end

function expected_information_item(latent::Latent1D, parameters::Parameters3PL)
    p = probability(latent.val, parameters)
    den = p * (1 - parameters.c)^2
    i_aa =                       (1 - p) * (p - parameters.c)   * (latent.latent.val - parameters.b)^2 / den
    i_ab = - parameters.a * (1 - p) * (p - parameters.c)^2 * (latent.val - parameters.b)   / den
    i_ac = i_aa                          * (p - parameters.c)   * (latent.val - parameters.b)   / den
    i_bc = - parameters.a * (1 - p) * (p - parameters.c)                                       / den
    i_bb = - parameters.a * i_bc    * (p - parameters.c)
    i_cc =                       (1 - p)                                                                 / den
    return [i_aa  i_ab  i_ac; i_ab  i_bb  i_bc; i_ac  i_bc  i_cc]
end

function observed_information_item(response_val::Float64, latent::latent1D, parameters::Parameters3PL)
    p = probability(latent.val, parameters)
    i = (1 - p) * (p - c)
    h = (response_val * parameters.c - p^2) * i
    j = response_val * i
    den = ((1 - parameters.c)*p)^2
    i_aa = - h * (latent.val - parameters.b)^2   / den
    i_ab = ((parameters.a * (latent.val - parameters.b) * h) + (p * (response_val - p) * (p - parameters.c))) / den
    i_ac = j * (latent.val - parameters.b) / den
    i_bc = parameters.a * j / den
    i_bb = - parameters.a^2 * h / den
    i_cc = ( response_val - 2*response_val*p + p^2) / den
    return [i_aa  i_ab  i_ac; i_ab  i_bb  i_bc; i_ac  i_bc  i_cc]
end

function observed_information_latent(response_val::Float64, latent::latent1D, parameters::Parameters3PL)
    p = probability(latent.val, parameters)
    i = (1 - p) * (p - c)
    h = (response_val * parameters.c - p^2) * i
    j = response_val * i
    den = ((1 - parameters.c)*p)^2
    i_aa = - h * (latent.val - parameters.b)^2   / den
    i_ab = ((parameters.a * (latent.val - parameters.b) * h) + (p * (response_val - p) * (p - parameters.c))) / den
    i_ac = j * (latent.val - parameters.b) / den
    i_bc = parameters.a * j / den
    i_bb = - parameters.a^2 * h / den
    i_cc = ( response_val - 2*response_val*p + p^2) / den
    return [i_aa  i_ab  i_ac; i_ab  i_bb  i_bc; i_ac  i_bc  i_cc]
end 
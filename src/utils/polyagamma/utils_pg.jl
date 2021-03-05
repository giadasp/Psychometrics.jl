struct PolyaGammaSample
    i_idx::Int64
    e_idx::Int64
    val::Float64
    PolyaGammaSample(i_idx, e_idx, val) = new(i_idx, e_idx, val)
end
#extract a random value from posterior and set it as value
function set_val_from_posterior!(item::AbstractItem; sampling = true)
    vals = _chain_append!(item.parameters; sampling = sampling)
    _set_val!(item.parameters, vals)
end

function set_val_from_posterior!(examinee::AbstractExaminee; sampling = true)
    val = _chain_append!(examinee.latent; sampling = sampling)
    _set_val!(examinee.latent, val)
end

#take the last value of the chain and set it as value
function set_val_from_chain!(item::AbstractItem)
    _set_val_from_chain!(item.parameters)
end

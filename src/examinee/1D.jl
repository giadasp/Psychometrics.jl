"""
Examinee1D <: AbstractExaminee

# Description
Examinee struct with a 1-dimensional latent variable.

# Fields
- **`idx::Int64`**: An integer that identifies the Examinee in this session.
- **`id::String`**: A string that identifies the Examinee.
- **`latent::Latent1D`**: A 1-dimensional latent variable associated with the Examinee.

# Factories
Examinee1D(idx, id, latent) = new(idx, id, latent)

Creates a new Examinee with custom index, id and 1-dimensional latent variable.

# Random initializers
Examinee1D(idx, id) = new(idx, id, Latent1D())

Randomly generates an Examinee with custom index and id and with a default 1-dimensional latent variable 
(Look at (`Latent1D`)[#Psychometrics.Latent1D] for the defaults).
"""
struct Examinee1D <: AbstractExaminee
idx::Int64
id::String
latent::Latent1D

# Factories
Examinee1D(idx, id, latent) = new(idx, id, latent)

# Random initializers
Examinee1D(id) = new(idx, id, Latent1D())
end


"""
    get_latents(examinee::Examinee1D)
"""
function get_latents(examinee::Examinee1D)
    [examinee.latent.val]
end

"""
    empty_chain!(examinee::Examinee1D)
"""
function empty_chain!(examinee::Examinee1D)
    examinee.latent.chain = Float64[]
end

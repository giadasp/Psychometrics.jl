module Psychometrics

import Distributions
import Dates

include("utils.jl")
include("dist.jl")
include("latent.jl")
include("parameters.jl")
include("item.jl")
include("examinee.jl")
include("design.jl")
include("response.jl")
include("latent.jl")
include("probability.jl")
include("likelihood.jl")
include("information.jl")

export add_prior!,add_posterior!,get_items,answer,get_examinee_by_idx,information_latent,informationi
end # module

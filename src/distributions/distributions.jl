import Random: GLOBAL_RNG
import Random
import SpecialFunctions: loggamma, gamma, beta
#import Random:rand

include("polyagamma/polyagamma.jl")
#include("truncated_normal/truncated_normal_2.jl")
include("gamma/gamma.jl")
include("truncated_gamma/truncated_gamma.jl")
include("truncated_inverse_gaussian/truncated_inverse_gaussian.jl")

function posterior_2pl_quick(
    parameters::Vector{Vector{Float64}},
    responses::Vector{Float64},
    X::Vector{Float64},
    W::Vector{Float64}
    )   
    return map( (x, w) ->  
            mapreduce( (pars, r) -> 
            begin
                if (r > 0)
                    _sig_c(pars[1]*(x - pars[2])) 
                else
                    _sig_cplus(pars[1]*(x - pars[2]))
                end
            end
            ,
            *,
            parameters,
            responses,
            )*w,
        X,
        W
        ) 
end

# function posterior_quick(
#     responses::Vector{Union{Missing, Float64}},
#     sig_phi::Matrix{Float64},
#     W::Vector{Float64}
#     )   
#     K = size(X, 1)
#     post = ones(Float64, K)
#     for k = 1:K
#             post_k = 1.0
#             for i in 1:size(responses, 1)
#                 if (responses[i] > 0)
#                     post_k *= sig_phi[i, k]
#                 else
#                     post_k *= 1 - sig_phi[i, k]
#                 end
#             end
#             post[k] = post_k * W[k]
#         end
#     return post::Vector{Float64}
# end


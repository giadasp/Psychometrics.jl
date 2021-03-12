function posterior_quick(
    parameters::Vector{Vector{Float64}},
    responses::Vector{Union{Missing, Float64}},
    X::Vector{Float64},
    W::Vector{Float64}
    )   
    return map( (x, w) ->  
            mapreduce( (pars, r) -> 
            begin
                if (r > 0)
                    _sig_c(pars[1]*(x - pars[2])) * w
                else
                    _sig_cplus(pars[1]*(x - pars[2])) * w
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


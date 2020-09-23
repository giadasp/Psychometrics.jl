
				maxLHMMLE!(responses, items, latent_posterior, X, W)

function maxLHMMLE!(responses::Vector{<:AbstractResponse},
    items::Dict{Int64,<:AbstractItem}
    latent_posterior::Distributions.UnivariateDistribution,
    bounds_latent::Vector{Float64}
    X::Vector{Float64},
    W::Vector{Float64},
    )
    I = length(items)
    
    sumpk=zeros(Float64,K,nItems)
    r1=similar(sumpk)
    posterior=compPostSimp(posterior,N,K,nItems,iIndex,responses,Wk,phi)
    LinearAlgebra.BLAS.gemm!('T', 'T', one(Float64), posterior, design, zero(Float64), sumpk)# sumpk KxI
    LinearAlgebra.BLAS.gemm!('T', 'T', one(Float64), posterior, responses, zero(Float64), r1)# r1 KxI
    nPar=size(bds.minPars,1)
    opt=NLopt.Opt(:LD_SLSQP,nPar)
    opt.lower_bounds = bds.minPars
    opt.upper_bounds = bds.maxPars
    opt.xtol_rel = io.xTolRel
    opt.maxtime = io.timeLimit
    opt.ftol_rel=  io.fTolRel
    #opt.maxeval=50
    Distributed.@sync Distributed.@distributed for i=1:nItems
        pars_i=max_i(X,sumpk[:,i],r1[:,i],parsStart[:,i],nPar,opt)
        if nPar==1
            parsStart[2,i]=copy(pars_i)
        else
            parsStart[:,i]=copy(pars_i)
        end
    end
    LinearAlgebra.BLAS.gemm!('N', 'N', one(Float64), X, parsStart, zero(Float64), phi)# phi=New_pars*X1', if A'*B then 'T', 'N'
    return parsStart::Matrix{Float64}, phi::Matrix{Float64}
end
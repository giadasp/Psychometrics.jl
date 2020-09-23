function compute_posterior(responses::Vector{<:AbstractResponse},
    items::Dict{Int64,<:AbstractItem},
    X::Vector{Float64},
    W::Vector{Float64})
    ephi = [mapreduce(x -> r probability(x, i.parameters), *, X) for i in values(items)]
	post_n = zeros(Float64, K)
    
    for n = 1:N
		for k = 1:K
			post_k = one(Float64)
			for i in iIndex[n]
				if r[i, n] > 0
					post_k *= ephione[k, i]
				else
					post_k *= ephizero[k, i]
				end
			end
				post_n[k] = copy(post_k)
		end
		post_n = (post_n .* Wk)
		exp_cd = sum(post_n)
		if exp_cd > typemin(Float64)
			post_n=post_n./exp_cd
		end
		post[n,:]=copy(post_n)
	end
	return post::Matrix{Float64}
end
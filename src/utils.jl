function _exp_c(x::Float64)
	ccall(:exp, Float64, (Float64,), x)
end

function _log_c(x::Float64)
	ccall(:log, Float64, (Float64,), x)
end

function _log1p_c(x::Float64)
	ccall(:log1p, Float64, (Float64,), x)
end

function _sig_c(x::Float64)
	1 / (1 + _exp_c(-x))
end

function _sig_cplus(x::Float64)
	1 / ( 1 + _exp_c(x))
end

function _gemmblas(A::Matrix{Float64}, B::Matrix{Float64}, ol_max::Matrix{Float64})
	sa = size(A)
	ol = copy(ol_max)
	ccall(("dgemm_64_", "libopenblas64_"), Cvoid,
	(Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64},
	Ref{Int64}, Ref{Float64}, Ptr{Float64}, Ref{Int64},
	Ptr{Float64}, Ref{Int64}, Ref{Float64}, Ptr{Float64},
	Ref{Int64}),
	'T', 'N', sa[2], sa[2],
	sa[1], 1.0, A, max(1, stride(A, 2)),
	B, max(1, stride(B, 2)), - one(Float64), ol,
	max(1, stride(ol, 2)))
	return ol::Matrix{Float64}
end

function _gemvblas(A::Matrix{Float64}, x::Vector{Float64}, cons::Vector{Float64}, sizex::Int64)
	ccall(("dgemv_64_", "libopenblas64_"), Cvoid, (Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Float64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ref{Float64}, Ptr{Float64}, Ref{Int64}), 'N', size(A, 1), sizex, 1.0, A, max(1, stride(A, 2)), x, 1, -1.0, cons, 1)
	return cons
end

function _gemvblasT(A::Matrix{Float64}, x::Vector{Float64}, sizeA2::Int64)
	cons = zeros(Float64, sizeA2)
	ccall(("dgemv_64_", "libopenblas64_"), Cvoid, (Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Float64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ref{Float64}, Ptr{Float64}, Ref{Int64}), 'T', size(A, 1), size(A, 2), 1.0, A, max(1, stride(A, 2)), x, 1, -1.0, cons, 1)
	return cons
end

function discretize(dist::Distributions.UnivariateDistribution; K = 61, bounds= [-6.0, 6.0])
	X = collect(range(bounds[1], length = K, stop = bounds[2]))
	W = pdf.(dist, X) / sum(pdf.(dist, X))
	return (X, W)
end
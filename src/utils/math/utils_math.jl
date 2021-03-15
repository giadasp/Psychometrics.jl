function _exp_c(x::Float64)
    return ccall(:exp, Float64, (Float64,), x)
end

function _log_c(x::Float64)
    return ccall(:log, Float64, (Float64,), x)
end

function _log_cosh_c(x::Float64)
    return ccall(:log, Float64, (Float64,), ccall(:cosh, Float64, (Float64,), x))
end

function _log1p_c(x::Float64)
    return ccall(:log1p, Float64, (Float64,), x)
end

function _sig_c(x::Float64)
    return 1 / (1 + _exp_c(-x))
end

function _sig_cplus(x::Float64)
    return 1 / (1 + _exp_c(x))
end

function _p1p(x::Float64)
    return x * (1 - x)::Float64
end

Base.copy(x::T) where {T} = T([getfield(x, k) for k ∈ fieldnames(T)]...)
# log1pexp(x::Real) = x < 18.0 ? _log1p_c(_exp_c(x)) : x < 33.3 ? x + _exp_c(-x) : oftype(_exp_c(-x), x)

# """
#     cosh(x) = (1+e⁻²ˣ)/(2e⁻ˣ)
#     so _log_cosh_c(x) = _log_c((1+e⁻²ˣ)) + x - _log_c(2)
#                   = x + log1pexp(-2x) - _log_c(2)
# """
# function logcosh(x::Real)
#     return x + log1pexp(-2x) - _log_c(2)
# end 
function cutR(
    x::Vector{Float64};
    start = "minimum",
    stop = "maximum",
    n_bins = 2,
    return_breaks = true,
    return_mid_points = false,
)
    if (start == "minimum")
        start = minimum(x)
    end
    if (stop == "maximum")
        stop = maximum(x)
    end
    bw = (stop - start) / (n_bins - 1)
    midPts = zeros(n_bins)
    for i = 1:n_bins
        midPts[i] = start + (i - 1) * bw
    end
    breaks = collect(range(start - (bw / 2); length = n_bins + 1, stop = stop + (bw / 2)))
    y = zeros(size(x, 1))
    for j = 1:size(x, 1)
        for i = 1:n_bins
            if (x[j] >= breaks[i]) && (x[j] < breaks[i+1])
                y[j] = i
            end
            if i == n_bins && x[j] == breaks[i+1]
                y[j] = i
            end
        end
    end
    if (return_breaks == true || return_mid_points == true)
        if return_mid_points == false
            return (Int.(y), breaks)
        elseif return_breaks == false
            return (Int.(y), midPts)
        else
            return (Int.(y), breaks, midPts)
        end
    else
        return Int.(y)
    end
end

function _gemmblasATB!(
    A::Matrix{Float64},
    B::Matrix{Float64},
    ret::Matrix{Float64},
    alpha::Float64,
    beta::Float64,
)
    sa = size(A)
    ccall(
        ("dgemm_64_", "libopenblas64_"),
        Cvoid,
        (
            Ref{UInt8},
            Ref{UInt8},
            Ref{Int64},
            Ref{Int64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
        ),
        'T',
        'N',
        sa[2],
        sa[2],
        sa[1],
        alpha,
        A,
        max(1, stride(A, 2)),
        B,
        max(1, stride(B, 2)),
        beta,
        ret,
        max(1, stride(ret, 2)),
    )
end

function _gemmblasATB(A::Matrix{Float64}, B::Matrix{Float64}, alpha::Float64, beta::Float64)
    sa = size(A)
    ret = zeros(Float64, sa[2], size(B, 2))
    ccall(
        ("dgemm_64_", "libopenblas64_"),
        Cvoid,
        (
            Ref{UInt8},
            Ref{UInt8},
            Ref{Int64},
            Ref{Int64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
        ),
        'T',
        'N',
        sa[2],
        sa[2],
        sa[1],
        alpha,
        A,
        max(1, stride(A, 2)),
        B,
        max(1, stride(B, 2)),
        beta,
        ret,
        max(1, stride(ret, 2)),
    )
    return ret::Matrix{Float64}
end

function _gemmblasAB(A::Matrix{Float64}, B::Matrix{Float64})
    sa = size(A)
    ret = zeros(Float64, sa[1], size(B, 2))
    ccall(
        ("dgemm_64_", "libopenblas64_"),
        Cvoid,
        (
            Ref{UInt8},
            Ref{UInt8},
            Ref{Int64},
            Ref{Int64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
        ),
        'N',
        'N',
        sa[1],
        size(B, 2),
        sa[2],
        one(Float64),
        A,
        max(1, stride(A, 2)),
        B,
        max(1, stride(B, 2)),
        one(Float64),
        ret,
        max(1, stride(ret, 2)),
    )
    return ret::Matrix{Float64}
end

function _gemvblasAx(
    A::Matrix{Float64},
    x::Vector{Float64},
    ret::Vector{Float64},
    sizex::Int64,
)
    ccall(
        ("dgemv_64_", "libopenblas64_"),
        Cvoid,
        (
            Ref{UInt8},
            Ref{Int64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
        ),
        'N',
        size(A, 1),
        sizex,
        1.0,
        A,
        max(1, stride(A, 2)),
        x,
        1,
        -1.0,
        ret,
        1,
    )
    return ret
end

function _gemvblasATx(A::Matrix{Float64}, x::Vector{Float64})
    ret = zeros(Float64, size(A, 2))
    ccall(
        ("dgemv_64_", "libopenblas64_"),
        Cvoid,
        (
            Ref{UInt8},
            Ref{Int64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ref{Float64},
            Ptr{Float64},
            Ref{Int64},
        ),
        'T',
        size(A, 1),
        size(A, 2),
        1.0,
        A,
        max(1, stride(A, 2)),
        x,
        1,
        -1.0,
        ret,
        1,
    )
    return ret
end

function discretize(
    dist::Distributions.UnivariateDistribution;
    K = 61,
    bounds = [-6.0, 6.0],
)
    X = collect(range(bounds[1], length = K, stop = bounds[2]))
    W = Distributions.pdf.(dist, X) / sum(Distributions.pdf.(dist, X))
    return (X, W)
end

function _matrix_cols_vec(A::Matrix{Float64}, v::Vector{Float64}, fun::Function)
    #r = similar(A)
    r = mapreduce(col -> map((x, y) -> fun(x, y), col, v), hcat, eachcol(A))
    # @inbounds for j = 1:size(A,2) 
    #     for i = 1:size(A,1) 
    #         r[i,j] = fun(A[i,j], v[j])
    #     end
    # end 
    r::Matrix{Float64}
end


function _matrix_rows_vec(A::Matrix{Float64}, v::Vector{Float64}, fun::Function)
    #r = similar(A)
    r = mapreduce(row -> map((x, y) -> fun(x, y), row, v), hcat, eachrow(A))
    # @inbounds for j = 1:size(A,2) 
    #     for i = 1:size(A,1) 
    #         r[i,j] = fun(A[i,j], v[j])
    #     end
    # end 
    r::Matrix{Float64}
end

Base.copy(x::T) where {T} = T([getfield(x, k) for k ∈ fieldnames(T)]...)
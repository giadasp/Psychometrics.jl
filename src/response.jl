"""
    AbstractResponse
"""
abstract type AbstractResponse end

"""
`Response <: AbstractResponse`

# Description

An immutable which holds the information about response, such as the identifier of the examinee who gave the response,
`examinee_id::String`, the identifier of the answered item `item_idx::String`, and the answer starting and ending times `start_time::Dates.DateTime` and `end_time::Dates.DateTime`.

"""
struct Response <: AbstractResponse
    item_idx::Int64
    examinee_idx::Int64
    item_id::String
    examinee_id::String
    val::Union{Missing,Float64}
    start_time::Dates.DateTime
    end_time::Dates.DateTime
    Response(
        item_idx,
        examinee_idx,
        item_id,
        examinee_id,
        val,
        start_time,
        end_time,
    ) = new(item_idx, examinee_idx, item_id, examinee_id, val, start_time, end_time)
    Response(
        item_idx,
        examinee_idx,
        item_id,
        examinee_id,
        val,
        time::Dates.DateTime,
    ) = new(item_idx, examinee_idx, item_id, examinee_id, val, time, time)
end

# Outer Constructor Methods

"""
```julia
get_examinees_by_item_id(
    item_id::String,
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
)
```

# Description

It returns the examinees who answered to the item with id `item_id`.
"""
function get_examinees_by_item_id(
    item_id::String,
    responses::Vector{<:AbstractResponse},
    examinees::Vector{<:AbstractExaminee},
)
    examinees[filter(r -> r.item_id == item_id, responses).examinee_id]
end

"""
```julia
get_items_by_examinee_id(
    examinee_id::String,
    responses::Vector{<:AbstractResponse},
    items::Vector{<:AbstractItem},
)
```

# Description

It returns the items answered by the examinee with id `examinee_id`.
"""
function get_items_by_examinee_id(
    examinee_id::String,
    responses::Vector{<:AbstractResponse},
    items::Vector{<:AbstractItem},
)
    items[filter(r -> r.examinee_id == examinee_id, responses).item_id]
end

"""
```julia
add_response!(response::AbstractResponse, responses::Vector{<:AbstractResponse})
```

# Description

Push the response in the response vector `responses`.
"""
function add_response!(response::AbstractResponse, responses::Vector{<:AbstractResponse})
    push!(response, responses)
end

"""
```julia
get_responses_by_examinee_id(examinee_id::String, responses::Vector{<:AbstractResponse})
```

# Description

It returns the vector of responses given by examinee with id = `id`.
"""
function get_responses_by_examinee_id(
    examinee_id::String,
    responses::Vector{<:AbstractResponse},
)
    filter(r -> r.examinee_id == examinee_id, responses)
end

"""
```julia
get_responses_by_item_id(item_id::String, responses::Vector{<:AbstractResponse})
```

# Description

It returns the vector of responses to item with id equal to `item_id`.
"""
function get_responses_by_item_id(item_id::String, responses::Vector{<:AbstractResponse})
    filter(r -> r.item_id == item_id, responses)
end

"""
```julia
get_responses_by_item_idx(item_idx::Int64, responses::Vector{<:AbstractResponse}; sorted = true)
```

# Description

It returns the vector of responses to item with idx equal to `item_idx`.
The vector of responses is sorted by `examinee_idx` if `sorted = true`.
"""
function get_responses_by_item_idx(
    item_idx::Int64,
    responses::Vector{<:AbstractResponse};
    sorted = true,
)
    resp_item = filter(r -> r.item_idx == item_idx, responses)
    if sorted
        sort!(resp_item, by = r -> r.examinee_idx)
    end
end

"""
```julia
_generate_response(latent::Latent1D, parameters::AbstractParametersBinary)
```

# Description

Randomly generate a response for a 1-dimensional latent variable and custom item parameters.
"""
function _generate_response(latent::Latent1D, parameters::AbstractParametersBinary)
    Float64(rand(Distributions.Bernoulli(_probability(
        latent,
        parameters,
        Array{Float64,1}(undef, 0),
        Array{Float64,1}(undef, 0),
    ))))::Float64
end

"""
```julia
answer(examinee::AbstractExaminee, item::AbstractItem)
```

# Description

Randomly generate a response by `examinee` to a dichotomous (binary) `item`.
"""
function answer(examinee::AbstractExaminee, item::AbstractItem)
    Response(
        item.idx,
        examinee.idx,
        item.id,
        examinee.id,
        _generate_response(examinee.latent, item.parameters),
        Dates.now(),
        Dates.now(),
    )
end


"""
```julia
answer(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
```

# Description

Randomly generate a response by `examinee` to dichotomous (binary) `items`.
"""
function answer(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
    map(
        i -> Response(
            i.idx,
            examinee.idx,
            i.id,
            examinee.id,
            _generate_response(examinee.latent, i.parameters),
            Dates.now(),
        ),
        items,
    )
end

"""
```julia
answer(examinee_id::String, item_id::String, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
```

# Description

Randomly generate a response by `Examinee` with index `examinee_id` to a dichotomous (binary) `item` with index `item_id`.
"""
function answer(
    examinee_id::String,
    item_id::String,
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItem},
)
    answer(
        get_examinee_by_id(examinee_id, examinees),
        get_item_by_id(item_id, items),
    )
end

"""
```julia
answer(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
```

# Description

Randomly generate responses by all the examinees in `examinees` to items in `items`.
"""
function answer(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
    mapreduce(e -> map(i -> answer(e, i), items), vcat, examinees)
end

"""
```julia
get_design_matrix(responses::Vector{Response}, I::Int64, N::Int64)
```

# Description

Returns the ``I \times N `` design matrix.
"""
function get_design_matrix(responses::Vector{Response}, I::Int64, N::Int64)
    has_answered = map(r -> CartesianIndex(r.item_idx, r.examinee_idx), responses)
    design = zeros(Float64, I, N)
    design[has_answered] .= one(Float64)
    return design::Matrix{Float64}
end

"""
```julia
get_response_matrix(responses::Vector{Response}, I::Int64, N::Int64)
```

# Description

Transform vector of `Response`s in a ``I \times N`` response matrix.
A non given answer has value `0.0`.
"""
function get_response_matrix(responses::Vector{Response}, I::Int64, N::Int64)
    response_matrix = Matrix{Union{Missing, Float64}}(missing .* ones(Float64, I, N))
    
    map(r -> response_matrix[CartesianIndex(r.item_idx, r.examinee_idx)] = r.val, responses)
    return response_matrix::Matrix{Union{Missing, Float64}}
end

"""
```julia
get_responses(response_matrix::Matrix{Float64}, design_matrix::Matrix{Float64}, items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})
```

Transforms a ``I \times N`` response matrix in a vector of `Response`s given a valid `design_matrix`, a vector of `Item`s and a vector of `Examinee`s.
"""
function get_responses(
    response_matrix::Matrix{Float64},
    design_matrix::Matrix{Float64},
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
)
    mapreduce(
        e -> map(
            i -> Response(
                i.idx,
                e.idx,
                i.id,
                e.id,
                response_matrix[i.idx, e.idx],
                Dates.now(),
            ),
            items[findall(design_matrix[:, e.idx] .> 0.0)],
        ),
        vcat,
        examinees,
    )
end

"""
```julia
get_items_idx_answered_by_examinee(
    examinee::AbstractExaminee,
    responses::Vector{<:AbstractResponse},
)
```

# Description

Returns the idx of the items answered by examinee.
"""
function get_items_idx_answered_by_examinee(
    examinee::AbstractExaminee,
    responses::Vector{<:AbstractResponse},
)
    resp_e = get_responses_by_examinee_id(examinee.id, responses)
    return map(r -> r.item_idx, resp_e)
end

"""
```julia
get_examinees_idx_who_answered_item(
    item::AbstractItem,
    responses::Vector{<:AbstractResponse},
)
```

# Description

Returns the idx of the examinees who answered to item.
"""
function get_examinees_idx_who_answered_item(
    item::AbstractItem,
    responses::Vector{<:AbstractResponse},
)
    resp_e = get_responses_by_item_id(item.id, responses)
    return map(r -> r.examinee_idx, resp_e)
end

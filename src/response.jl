abstract type AbstractResponse end

struct ResponseBinary <: AbstractResponse
    item_idx::Int64
    examinee_idx::Int64
    item_id::String
    examinee_id::String
    val::Union{Missing, Float64}
    start_time::Dates.DateTime
    end_time::Dates.DateTime
    ResponseBinary(item_idx, examinee_idx, item_id, examinee_id, val, start_time, end_time) =
        new(item_idx, examinee_idx, item_id, examinee_id, val, start_time, end_time)
    ResponseBinary(item_idx, examinee_idx, item_id, examinee_id, val, time::Dates.DateTime) =
        new(item_idx, examinee_idx, item_id, examinee_id, val, time, time)
end

# Outer Constructor Methods

"""
    get_examinees_by_item_id(item_id::String, responses::Vector{<:AbstractResponse}, examinees::Vector{<:AbstractExaminee})

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
    get_items_by_examinee_id(examinee_id::String, responses::Vector{<:AbstractItem})

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
    add_response!(response::AbstractResponse, responses::Vector{<:AbstractResponse})

Push the response in the response vector `responses`.
"""
function add_response!(response::AbstractResponse, responses::Vector{<:AbstractResponse})
    push!(response, responses)
end

"""
    get_responses_by_examinee_id(examinee_id::String, responses::Vector{<:AbstractResponse})

It returns the vector of responses given by examinee with id = `id`.
"""
function get_responses_by_examinee_id(examinee_id::String, responses::Vector{<:AbstractResponse})
    filter(r -> r.examinee_id == examinee_id, responses)
end

"""
    get_responses_by_item_id(item_id::String, responses::Vector{<:AbstractResponse})

It returns the vector of responses to item with id equal to `item_id`.
"""
function get_responses_by_item_id(item_id::String, responses::Vector{<:AbstractResponse})
    filter(r -> r.item_id == item_id, responses)
end

"""
    get_responses_by_item_idx(item_idx::Int64, responses::Vector{<:AbstractResponse}; sorted = true)

It returns the vector of responses to item with idx equal to `item_idx`.
The vector of responses is sorted if `sorted = true`.
"""
function get_responses_by_item_idx(
    item_idx::Int64,
    responses::Vector{<:AbstractResponse};
    sorted = true
)
    resp_item = filter(r -> r.item_idx == item_idx, responses)
    if sorted
        sort!(resp_item, by = r -> r.examinee_idx)
    end
end

"""
    generate_response(latent::Latent1D, parameters::AbstractParametersBinary)

Randomly generate a response for a 1-dimensional latent variable and custom item parameters.
"""
function generate_response(latent::Latent1D, parameters::AbstractParametersBinary)
    Float64(rand(Distributions.Bernoulli(probability(
        latent,
        parameters,
        Array{Float64,1}(undef, 0),
        Array{Float64,1}(undef, 0),
    ))))::Float64
end

"""
<<<<<<< HEAD
    answer(examinee::AbstractExaminee, item::AbstractItem)
=======
    answer(examinee::AbstractExaminee, item::AbstractItemBinary)
>>>>>>> release

Randomly generate a dichotomous (binary) response by `examinee` to a dichotomous (binary) `item`.
"""
<<<<<<< HEAD
function answer(examinee::AbstractExaminee, item::AbstractItem)
    Response(
=======
function answer(examinee::AbstractExaminee, item::AbstractItemBinary)
    ResponseBinary(
>>>>>>> release
        item.idx,
        examinee.idx,
        item.id,
        examinee.id,
        generate_response(examinee.latent, item.parameters),
        Dates.now(),
    )
end

"""
<<<<<<< HEAD
    answer(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
=======
    answer(examinee::AbstractExaminee, items::Vector{<:AbstractItemBinary})
>>>>>>> release

Randomly generate a dichotomous (binary) response by `examinee` to dichotomous (binary) `items`.
"""
<<<<<<< HEAD
function answer(examinee::AbstractExaminee, items::Vector{<:AbstractItem})
    map(
        i -> Response(
=======
function answer(examinee::AbstractExaminee, items::Vector{<:AbstractItemBinary})
    map(
        i -> ResponseBinary(
>>>>>>> release
            i.idx,
            examinee.idx,
            i.id,
            examinee.id,
            generate_response(examinee.latent, i.parameters),
            Dates.now(),
        ),
        items,
    )
end

"""
<<<<<<< HEAD
    answer(examinee_id::String, item_id::String, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})

Randomly generate a response by `Examinee` with index `examinee_id` to `item` with index `item_id`.
=======
    answer(examinee_id::String, item_id::String, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItemBinary})

Randomly generate a dichotomous (binary) response by `Examinee` with index `examinee_id` to a dichotomous (binary) `item` with index `item_id`.
>>>>>>> release
"""
function answer(
    examinee_id::String,
    item_id::String,
    examinees::Vector{<:AbstractExaminee},
    items::Vector{<:AbstractItemBinary},
)
    answer(get_examinee_by_id(examinee_id, examinees), get_item_by_id(item_id, items))
end

"""
<<<<<<< HEAD
    answer(examinees:Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})
=======
    answer(examinees:Vector{<:AbstractExaminee}, items::Vector{<:AbstractItemBinary})
>>>>>>> release

Randomly generate dichotomous (binary) responses by all the examinees in `examinees` to dichotomous (binary) items in `items`.
"""
function answer(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItemBinary})
    mapreduce(e -> map(i -> answer(e, i), items), vcat, examinees)
end

"""
    get_design_matrix(responses::Vector{ResponseBinary}, I::Int64, N::Int64)

Returns the dichotomous (binary) `I x N` design matrix.
"""
function get_design_matrix(responses::Vector{ResponseBinary}, I::Int64, N::Int64)
    has_answered = map(r -> CartesianIndex(r.item_idx, r.examinee_idx), responses)
    design = zeros(Float64, I, N)
    design[has_answered] .= one(Float64)
    return design::Matrix{Float64}
end

"""
    get_response_matrix(responses::Vector{ResponseBinary}, I::Int64, N::Int64)

Transform vector of `ResponseBinary`s in a `I x N` response matrix.
A non given answer has value `0.0`.
"""
function get_response_matrix(responses::Vector{ResponseBinary}, I::Int64, N::Int64)
    response_matrix = zeros(Float64, I, N)
    map(r -> response_matrix[CartesianIndex(r.item_idx, r.examinee_idx)] = r.val, responses)
    return response_matrix::Matrix{Float64}
end


"""
    get_responses(response_matrix::Matrix{Float64}, design_matrix::Matrix{Float64}, items::Vector{<:AbstractItem}, examinees::Vector{<:AbstractExaminee})

Transforms a `I x N` response matrix in a vector of `ResponseBinary`s given a valid `design_matrix`, a vector of `Item`s and a vector of `Examinee`s.
"""
function get_responses(
    response_matrix::Matrix{Float64},
    design_matrix::Matrix{Float64},
    items::Vector{<:AbstractItem},
    examinees::Vector{<:AbstractExaminee},
)
    mapreduce(
        e -> map(
            i -> ResponseBinary(
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


function get_items_idx_answered_by_examinee(
    examinee::AbstractExaminee,
    responses::Vector{<:AbstractResponse},
)
    resp_e = get_responses_by_examinee_id(examinee.id, responses)
    return map(r -> r.item_idx, resp_e)
end

function get_examinees_idx_who_answered_item(
    item::AbstractItem,
    responses::Vector{<:AbstractResponse},
)
    resp_e = get_responses_by_item_id(item.id, responses)
    return map(r -> r.examinee_idx, resp_e)
end


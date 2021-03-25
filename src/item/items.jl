
"""
    get_parameters_vals(items::Vector{<:AbstractItem})

# Description

Returns a matrix with item parameters displayed by row.
"""
function get_parameters_vals(items::Vector{<:AbstractItem})
ret = Vector{Vector{Float64}}(undef, size(items, 1))
max_length = 1
i_2 = 0
for i in items
    i_2 += 1
    local pars = get_parameters_vals(i)
    ret[i_2] = copy(pars)
    max_length = max_length < size(pars, 1) ? size(pars, 1) : max_length
end

for i_3 = 1:i_2
    local length_i = size(ret[i_3], 1)
    if length_i < max_length
        ret[i_3] = vcat(ret[i_3], zeros(Float64, max_length - length_i))
    end
end
return permutedims(reduce(hcat, ret))
end

"""
    get_parameters(items::Vector{<:AbstractItem})

# Description

Returns a vector of `<:AbstractParameters` objects.
"""
function get_parameters(items::Vector{<:AbstractItem})
return map( i -> i.parameters, items)
end

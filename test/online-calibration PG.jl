using Pkg
Pkg.activate(".")
using Psychometrics
using Distributions
using LinearAlgebra
using Dates
using Random

const I = 350
const N = 500
const test_length = 40
const iter_mcmc = 4000
# ITEM PARAMETERS AND LATENTS 

items = [Item2PL(i, string("item_", i), ["math"], Parameters2PL(Product([LogNormal(0.2,0.3), Normal(0,1)]), [1e-5,Inf], [-Inf, Inf])) for i = 1:I];
examinees = [Examinee1D(e, string("examinee_", e), Latent1D()) for e = 1:N];

# RESPONSES

responses = generate_response(examinees, items);

#INITIAL VALUES

items_est = [Item2PL(i, string("item_",i), ["math"], Parameters2PL(Product([TruncatedNormal(1.0, 1.0, 0.0, Inf), Normal(0,1)]), [1e-5,5.0], [-6.0, 6.0])) for i = 1 : I];
examinees_est = [Examinee1D(e, string("examinee_",e), Latent1D(Normal(0.0,1), [-6.0, 6.0])) for e = 1 : N]; 
map(e -> begin e.latent.val = 0.0 end, examinees_est)
#START ONLINE-CALIBRATION

function find_best_item(examinee::Examinee1D, items::Vector{<:AbstractItem})
    infos = information_latent(examinee, items)
    return items[findmax(infos)[2]].idx
end

items_per_examinee = Vector{Vector{Int64}}(undef,N)
responses_per_examinee = Vector{Vector{Response}}(undef,N)

for e in examinees_est
    println("true theta= ", examinees[e.idx].latent.val)
    items_e_idx = Int64[]
    responses_e = Response[]
    sorted_items_e_idx = Int64[]
    for it in 1:test_length
        #find best item
        next_i_idx= find_best_item(e, items_est[setdiff(collect(1:I), items_e_idx)])
        push!(items_e_idx, next_i_idx)
        sorted_items_e_idx = sort(items_e_idx)
        #answer (true theta and true item parameters)
        resp = generate_response(examinees[e.idx], items[next_i_idx])
        push!(responses_e, resp)
        sort!(responses_e, by = r -> r.item_idx)

        #estimate
        for iter in 1:iter_mcmc
            if mod(iter,100)==0 
               # println(iter)
            end
            #generate polyagammas
            W = generate_w(items_est[sorted_items_e_idx], e)
            if it > 37
                #MCMC on answered items
                map( i -> begin 
                    mcmc_iter!(i, [e], responses_e, map( y -> y.val, W); sampling = true)
                    end, 
                items_est[sorted_items_e_idx])
            end
            #MCMC on examinee
            mcmc_iter!(e, items_est[sorted_items_e_idx], responses_e, map( y -> y.val, W); sampling = true)    
        end
        #update examinee latent estimate
        update_estimate!(e)
        println("est theta at item ", it, " = ", e.latent.val)
    end
    #update item parameters estimate
    map( i -> begin 
        update_estimate!(i)
        end, 
    items_est[sorted_items_e_idx])

    #save items and responses
    items_per_examinee[e.idx] = sorted_items_e_idx
    responses_per_examinee[e.idx] = responses_e
    
    println("End examinee ", e.idx)
end

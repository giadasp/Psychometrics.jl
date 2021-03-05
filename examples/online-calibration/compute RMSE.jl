 using Pkg
 Pkg.instantiate()
 Pkg.activate(".")
 using Psychometrics
 using Distributions
 using LinearAlgebra
 using Dates
 using Random
 using SharedArrays
 using JLD2
 using StatsBase
 using DelimitedFiles
 using FileIO
reps=100
folder="test/simulation/batch50 NT400 Dgain"
items = load(string(folder,"/true values/true_items.jld2"), "items")

a_s = zeros(I_field,reps);
b_s = zeros(I_field,reps);
for rep = 1:reps
    # if isfile(string("test/online-calibration/rep_", rep, "_items_est.jld2"))
        # @load string("test/online-calibration/rep_", rep, "_items_est.jld2") items_est
        # a = map(i -> i.parameters.a, items_est)
        # a_s[:, rep] = copy(a)
        # b = map(i -> i.parameters.b, items_est)
        # b_s[:, rep] = copy(b)
    # else
        @load string(folder,"/rep_", rep, "_items_est_field.jld2") items_est_field
        a = map(i -> i.parameters.a, items_est_field)
        a_s[:, rep] = copy(a)
        b = map(i -> i.parameters.b, items_est_field)
        b_s[:, rep] = copy(b)
    #end
    
end

a_true = map(i -> i.parameters.a, items[(I_operational + 1) : I_total]);
a_true = hcat([a_true for rep = 1:reps]...);
b_true = map(i -> i.parameters.b, items[(I_operational + 1) : I_total]);
b_true = hcat([b_true for rep=1:reps]...);
diff_a = (a_s - a_true);
diff_b = (b_s - b_true);

using CSV
using DataFrames
a_true = map(i -> i.parameters.a, items[(I_operational + 1) : I_total]);
b_true = map(i -> i.parameters.b, items[(I_operational + 1) : I_total]);

RMSE_df = DataFrame(a_true = Vector{Float64}(a_true), b_true = Vector{Float64}(b_true), a_RMSE=[sqrt(sum(r.^2)/reps) for r in eachrow(diff_a)],b_RMSE=[sqrt(sum(r.^2)/reps) for r in eachrow(diff_b)]);
CSV.write(
    string(folder,"/RMSE.csv"),
    RMSE_df
)

CSV.write(
    string(folder,"/BIAS.csv"),
    DataFrame(a_BIAS=[sum(r.^2)/reps for r in eachrow(diff_a)],
    b_BIAS=[sum(r)/reps for r in eachrow(diff_b)])
)
using Plots
p = plot(layout=(5,5));
p_a = plot(ylim=(0,maximum(RMSE_df.a_RMSE)+0.15), xticks = (round.(unique(b_true), digits=2), string.(round.(unique(b_true), digits=2))), xlabel= "b", legend= :best);
global i=1
for a in unique(a_true)
    p_a= plot!(p_a, RMSE_df[RMSE_df.a_true .== a,:].b_true, RMSE_df[RMSE_df.a_true .== a,:].a_RMSE, label="a = " .*string.(round(a, digits=2)));
    i+=1
end
plot(p_a)
savefig(p_a,string(folder,"/a_RMSE.pdf"))
p_b = plot(ylim=(0,0.25), xticks = (round.(unique(a_true), digits=2), string.(round.(unique(a_true), digits=2))), xlabel= "a", legend= :best);
global i=1
for b in unique(b_true)
    p_b= plot!(p_b, RMSE_df[RMSE_df.b_true .== b,:].a_true, RMSE_df[RMSE_df.b_true .== b,:].b_RMSE, label="b = " .*string.(round(b, digits=2)));
    i+=1
end
plot(p_b)
savefig(p_b,string(folder,"/b_RMSE.pdf"))

println(sqrt(sum(diff_a.^2)/I_field/reps))
println(sqrt(sum(diff_b.^2)/I_field/reps))
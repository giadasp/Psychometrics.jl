
reps=100
folder="test/simulation/batch50  NT200 Dgain"
a_s = zeros(I_field,reps)
b_s = zeros(I_field,reps)
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
items = load(string(folder,"/true values/true_items.jld2"), "items")

a_true = map(i -> i.parameters.a, items[(I_operational + 1) : I_total])
a_true = hcat([a_true for rep = 1:reps]...)
b_true = map(i -> i.parameters.b, items[(I_operational + 1) : I_total])
b_true = hcat([b_true for rep=1:reps]...)
diff_a = (a_s - a_true)
diff_b = (b_s - b_true)

using CSV
using DataFrames
CSV.write(
    string(folder,"/RMSE.csv"),
    DataFrame(a_RMSE=[sqrt(sum(r.^2)/reps) for r in eachrow(diff_a)],
    b_RMSE=[sqrt(sum(r.^2)/reps) for r in eachrow(diff_b)])
)

CSV.write(
    string(folder,"/BIAS.csv"),
    DataFrame(a_BIAS=[sum(r.^2)/reps for r in eachrow(diff_a)],
    b_BIAS=[sum(r)/reps for r in eachrow(diff_b)])
)

println(sqrt(sum(diff_a)/I_field/reps))
println(sqrt(sum(diff_b)/I_field/reps))
chains = Vector{Vector{Vector{Float64}}}(undef, I_field)
folder="test/simulation/batch5 NT200 Dgain"

for i in 1 : I_field
    chains[i] = Vector{Vector{Float64}}(undef, 2)
    chains[i][1] = Vector{Float64}(undef, 0)
    chains[i][2] = Vector{Float64}(undef, 0)
    for resp in batch_size:batch_size:N_T
        @load string(folder,"/rep_1_item_", I_operational + i, "_resp_", resp, "_chain.jld2") chain
        r = Int64(resp/batch_size)
        for iter in 1:iter_mcmc_item
            push!(chains[i][1],chain[iter][1])
            push!(chains[i][2],chain[iter][2])
        end
    end
end

using Plots
plots=Vector{Plots.Plot}(undef, 5)
ma_window=1000
step_size = 1
starting_sample = Int64(iter_mcmc_item * (N_T/batch_size - 3 ))
for p in 1:5
    plots_p = Vector{Plots.Plot}(undef, 2)
    item=p+20

ma_a_1 =[sum(@view chains[item][1][i:(i+ma_window-1)])/ma_window for i in 1:(length(chains[item][1])-(ma_window-1))][starting_sample:step_size:end]
plots_p[1]=plot(ma_a_1);
plot!(plots_p[1], chains[item][1][(ma_window+starting_sample):step_size:end], alpha=0.2);
vline!(plots_p[1], ((collect((iter_mcmc_item-ma_window):(iter_mcmc_item):(iter_mcmc_item * N_T/batch_size)-starting_sample))./step_size) .+1  );
hline!(plots_p[1], [items[I_operational+item].parameters.a], ylim = (0,4) )

ma_b_1 =[sum(@view chains[item][2][i:(i+ma_window-1)])/ma_window for i in 1:(length(chains[item][2])-(ma_window-1))][starting_sample:step_size:end]
plots_p[2]=plot(ma_b_1);
plot!(plots_p[2], chains[item][2][(ma_window+starting_sample):step_size:end], alpha=0.2);
vline!(plots_p[2],collect((iter_mcmc_item-ma_window+starting_sample):iter_mcmc_item:(iter_mcmc_item * N_T/batch_size))./step_size);
hline!(plots_p[2],[items[I_operational+item].parameters.b], ylim = (-3,3));
plots[p] = plot(plots_p..., layout = (2, 1));
end

savefig(plot(plots..., layout = (1,5), legend = false, size=(1600,1200)),string(folder,"/file.pdf"))
plot(plots..., layout = (1,3), legend = false)

using Statistics,Distributions,Plots

function stationary_bootstrap_correlation(data1::Vector, data2::Vector, block_length::Int, n_bootstrap::Int, alpha::Float64)
    # Calculate correlation coefficient from original data 
      
      original_correlation = cor(data1, data2)
    
    # Calculate number of blocks
    n_blocks = length(data1) ÷ block_length
    
    # Bootstrap resampling
    bootstrap_correlations = Float64[]
    for _ in 1:n_bootstrap
        # Generate bootstrap indices for each block
        block_indices = []
        for _ in 1:n_blocks
            push!(block_indices, rand(1:length(data1) - block_length + 1))
        end
        
        # Resample with replacement
        resampled_data1 = [data1[block_indices[i]:block_indices[i] + block_length - 1] for i in 1:n_blocks]
        resampled_data2 = [data2[block_indices[i]:block_indices[i] + block_length - 1] for i in 1:n_blocks]
        
        # Calculate correlation coefficient from resampled data
        correlation = cor(vcat(resampled_data1...), vcat(resampled_data2...))
        push!(bootstrap_correlations, correlation)
    end
    
    # Sort bootstrap correlations
    sort!(bootstrap_correlations)
    
    # Calculate confidence intervals
    lower_index = round(Int, (alpha / 2) * n_bootstrap)
    upper_index = round(Int, (1 - alpha / 2) * n_bootstrap)
    lower_ci = bootstrap_correlations[lower_index]
    upper_ci = bootstrap_correlations[upper_index]
    
    Lcor = quantile(bootstrap_correlations, alpha/2)
    Ucor = quantile(bootstrap_correlations, 1-alpha/2)

    
#    return original_correlation, lower_ci, upper_ci
     
     println("sample correlation $original_correlation","lower_ci= $lower_ci", "upper_ci = $upper_ci")
     println("Lcor_quantile = $Lcor", "Ucor_quantile = $Ucor")
    
     histogram(bootstrap_correlations, bins=100, c=:blue,
     normed=true, opacity=0.4, label="Sample corrs")
     plot!([Lcor, Lcor],[0,50], c=:black, ls=:dash, label="$alpha% CI")
     plot!([Ucor, Ucor],[0,50],c=:black, ls=:dash, label="",
     xlabel="Sample Corrs", ylabel="Density")
end

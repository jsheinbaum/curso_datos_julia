using Statistics

function bootstrap_correlation(data1::Vector, data2::Vector, n_bootstrap::Int, alpha::Float64)
    # Calculate correlation coefficient from original data
    original_correlation = cor(data1, data2)
    
    # Bootstrap resampling
    bootstrap_correlations = Float64[]
    n = length(data1)
    for _ in 1:n_bootstrap
        # Resample with replacement
        indices = rand(1:n, n)
        resampled_data1 = data1[indices]
        resampled_data2 = data2[indices]
        
        # Calculate correlation coefficient from resampled data
        correlation = cor(resampled_data1, resampled_data2)
        push!(bootstrap_correlations, correlation)
    end
    
    # Sort bootstrap correlations
    sort!(bootstrap_correlations)
    
    # Calculate confidence intervals
    lower_index = round(Int, (alpha / 2) * n_bootstrap)
    upper_index = round(Int, (1 - alpha / 2) * n_bootstrap)
    lower_ci = bootstrap_correlations[lower_index]
    upper_ci = bootstrap_correlations[upper_index]
    
    return original_correlation, lower_ci, upper_ci
end

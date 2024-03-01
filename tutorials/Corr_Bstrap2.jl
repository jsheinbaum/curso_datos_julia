using Statistics, Distributions, Plots

function bootstrap_correlation2(data1::Vector, data2::Vector, n_bootstrap::Int, alpha::Float64)
    # Calculate correlation coefficient from original data original_correlation = cor(data1, data2)

    n=length(data1)
    corsample=cor(data1,data2)

    # Bootstrap resampling
     bootstrapCorrs = Float64[]
     for _ in 1:n_bootstrap
     indices = rand(1:n, n)
#bootstrapCorrs = [cor(rand(data1, n),rand(data2, n)) for i in 1:n_bootstrap]
     Corrs = cor(data1[indices],data2[indices]) 
     push!(bootstrapCorrs,Corrs)
     end
    Lcor = quantile(bootstrapCorrs, alpha/2)
    Ucor = quantile(bootstrapCorrs, 1-alpha/2)

    bootstrapSampleMedians = [median(rand(bootstrapCorrs, n)) for i in 1:n_bootstrap]
    Lmed = quantile(bootstrapSampleMedians, alpha/2)
    Umed = quantile(bootstrapSampleMedians, 1-alpha/2)

    println("correlation sample data: ", corsample )
    println("Bootstrap confidence interval for the correlation: ", (Lcor, Ucor) )
    println("Bootstrap confidence interval for the median: ", (Lmed, Umed) )

#    stephist(bootstrapCorrs, bins=100, c=:blue,
#    normed=true, label="Sample corrs")

    histogram(bootstrapCorrs, bins=100, c=:blue,
    normed=true, opacity=0.4, label="Sample corrs")
    
    plot!([Lcor, Lcor],[0,50], c=:black, ls=:dash, label="$alpha% CI")
    plot!([Ucor, Ucor],[0,50],c=:black, ls=:dash, label="",
    xlabel="Sample Corrs", ylabel="Density")
end

using Plots
using LinearAlgebra

    function plot_ellipse_from_covariance_matrix2(u,v; center=(0, 0),n_points=100, factor, kwargs...)
    # Compute eigenvalues and eigenvectors of the covariance matrix
    kz=[u v];
    m,n=size(kz)
    if n > m
    kz=kz';
    end
    cov_matrix=cov(kz)
    println("cov_mat ",cov_matrix)
    eigenvalues, eigenvectors = eigen(cov_matrix)
    
    # Find the indices of the largest and smallest eigenvalues
    max_index = argmax(eigenvalues)
    min_index = argmin(eigenvalues)
    
    # Extract the corresponding eigenvectors
    major_axis = eigenvectors[:, max_index]
    minor_axis = eigenvectors[:, min_index]
    
    # Compute the lengths of the semimajor and semiminor axes
    major_length = sqrt(eigenvalues[max_index])
    minor_length = sqrt(eigenvalues[min_index])
    
    # Generate parameter values for angle θ
    θ_values = LinRange(0, 2π, n_points)
    
    # Compute x and y coordinates of the ellipse
	
    pplot=Plots.scatter(u,v,aspect_ratio=:equal, xlabel="x", ylabel="y", legend=false; kwargs...)
    for i=1:length(factor)
	x_values = [center[1] + factor[i] * major_length * cos(θ) * major_axis[1] + factor[i] * minor_length * sin(θ) * minor_axis[1] for θ in θ_values]
	y_values = [center[2] + factor[i] * major_length * cos(θ) * major_axis[2] + factor[i] * minor_length * sin(θ) * minor_axis[2] for θ in θ_values]
    
    # Plot the ellipse
    pplot=Plots.plot!(x_values, y_values, aspect_ratio=:equal, xlabel="x", ylabel="y", legend=false; kwargs...)
   end
   Plots.plot(pplot)
end


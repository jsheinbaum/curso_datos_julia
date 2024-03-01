using Plots
using LinearAlgebra

    function get_ellipse_from_covariance_matrix(cov_matrix; center=(0, 0),n_points=100, factor)
    # Compute eigenvalues and eigenvectors of the covariance matrix
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
    x_values = [center[1] + factor[i] * major_length * cos(θ) * major_axis[1] + factor[i] * minor_length * sin(θ) * minor_axis[1] for i in factor, θ in θ_values]
    y_values = [center[2] + factor[i] * major_length * cos(θ) * major_axis[2] + factor[i] * minor_length * sin(θ) * minor_axis[2] for i in factor, θ in θ_values]
    
    # Plot the ellipse
#      Plots.plot!(x_values, y_values, aspect_ratio=:equal, xlabel="x", ylabel="y", legend=false; kwargs...)
      return (x_values,y_values)
end


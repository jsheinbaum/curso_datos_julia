using Plots

function plot_ellipse(a, b; θ_range=(0, 2π), n_points=100)
    # Generate parameter values for angle θ
    θ_values = LinRange(θ_range[1], θ_range[2], n_points)
    
    # Compute x and y coordinates of the ellipse
    x_values = a * cos.(θ_values)
    y_values = b * sin.(θ_values)
    
    # Plot the ellipse
    plot!(x_values, y_values, aspect_ratio=:equal, xlabel="x", ylabel="y", legend=false)
end

# Example usage:
#a = 2   # Semimajor axis
#b = 1   # Semiminor axis
#plot_ellipse(a, b)

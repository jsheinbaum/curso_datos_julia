
#using Plots
function plot_vector!(p1::Tuple, p2::Tuple,xlimits::Tuple,ylimits::Tuple)
    # Extract coordinates of the points
    x1, y1 = p1
    x2, y2 = p2
#    xlims = xlimits
#    ylims = ylimits
    # Compute components of the vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Create a plot with arrows representing the vector
    plot([x1, x2], [y1, y2], arrow=true, legend=false, color=:blue, xlabel="x", ylabel="y", xlims=xlimits, ylims=ylimits)
    annotate!([(x2, y2, text("($dx, $dy)", 8, :left))])
end
